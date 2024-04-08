import os
import time
import logging
import datetime


import sys
import torch


import torch.utils.data
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from fastai.vision import *

from Dino.utils.utils import Logger

from Dino.modules import utils
from Dino.loss.Dino_loss import DINOLoss
from torchvision import models as torchvision_models
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

from src.dataset import _get_databaunch
from src.engine import train_one_epoch
from src.model import build_model
from src.option import _parse_arguments


def train(config):
    """parameter configuration"""
    utils.init_distributed_mode(config)
    utils.fix_random_seeds(config.seed)
    cudnn.benchmark = True

    """dataset preparation"""
    logging.info('Construct dataset.')
    train_dataloader = _get_databaunch(config)
    config.iter_num = len(train_dataloader)
    logging.info(f"each epoch iteration: {config.iter_num}")

    student, teacher, teacher_without_ddp = build_model(config)

    """ setup loss """
    config.epochs = int(config.training_epochs * len(train_dataloader) * (
                config.batch_size_per_gpu * utils.get_world_size()) / config.imgnet_based) + 1
    
    print(f'training epochs is {config.epochs}')
    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        config.out_dim,
        config.crops_number,
        config.warmup_teacher_temp,
        config.teacher_temp,
        config.warmup_teacher_temp_epochs,
        config.epochs,
    ).cuda()
    
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif config.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    
    # for mixed precision training
    fp16_scaler = None
    if config.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    print('config info')
    for attr in ['lr', 'batch_size_per_gpu', 'min_lr', 'training_epochs', 'warmup_epoch', 'imgnet_based']:
        print(getattr(config, attr))

    lr_schedule = utils.cosine_iter_scheduler(
        base_value=config.lr * (config.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        final_value=config.min_lr,
        niter=config.training_epochs * len(train_dataloader),
        warmup_iters=int(
            (config.warmup_epoch * config.imgnet_based) / (config.batch_size_per_gpu * utils.get_world_size())),
    )
    
    wd_schedule = utils.cosine_iter_scheduler(
        config.weight_decay,
        config.weight_decay_end,
        config.training_epochs * len(train_dataloader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_iter_scheduler(config.momentum_teacher, 1,
                                                    config.training_epochs * len(train_dataloader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, 'iteration': 0}
    utils.restart_from_checkpoint(
        os.path.join(config.output_dir, config.global_name, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )

    iteration = int(to_restore["iteration"])
    epoch = to_restore["epoch"]
    print(f'continue to train:{iteration}:{epoch}')
    start_time = time.time()
    global global_epoch
    global_epoch = 0

    print("Starting DINO training !")
    for train_epoch in range(config.training_epochs):
        iteration = train_one_epoch(
            config,
            train_dataloader,
            train_epoch,
            student,
            teacher,
            teacher_without_ddp,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            dino_loss,
            fp16_scaler,
            iteration,
            global_epoch
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    config = _parse_arguments()
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    # _set_random_seed(config.global_seed)
    logging.info(config)
    os.makedirs(f"./saved_models/{config.global_name}", exist_ok=True)
    os.makedirs(f"./tensorboard", exist_ok=True)
    config.writer = SummaryWriter(log_dir=f"./tensorboard/{config.global_name}")
    train(config)