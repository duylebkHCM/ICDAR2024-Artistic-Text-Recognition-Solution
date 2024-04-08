import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed as dist

import torch.utils.data
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from fastai.vision import *

from Dino.utils.utils import Config, Logger, MyConcatDataset
from Dino.utils.util import Averager
from Dino.dataset.datasetsupervised_kmeans import ImageDatasetSelfSupervisedKmeans
from Dino.dataset.dataset import collate_fn_filter_none

from Dino.modules.vision_transformer import DINOHead
from Dino.modules import vision_transformer as vits
from Dino.modules import segmentor
from Dino.model.dino_vision import ABIDINOModel
from Dino.modules import utils
from Dino.loss.Dino_loss import DINOLoss
from torchvision import models as torchvision_models
from Dino.modules import svtr as svtr

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

def build_model(config):
    """ model configuration """
    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    config.arch = config.arch.replace("deit", "vit")
    
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if config.arch in vits.__dict__.keys():
        student = vits.__dict__[config.arch](
            patch_size=config.patch_size,
            drop_path_rate=config.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[config.arch](patch_size=config.patch_size)
        embed_dim = student.embed_dim
        
    # # if the network is a XCiT
    # elif config.arch in torch.hub.list("facebookresearch/xcit:main"):
    #     student = torch.hub.load('facebookresearch/xcit:main', config.arch,
    #                              pretrained=False, drop_path_rate=config.drop_path_rate)
    #     teacher = torch.hub.load('facebookresearch/xcit:main', config.arch, pretrained=False)
    #     embed_dim = student.embed_dim
        
    # otherwise, we check if the architecture is in torchvision models
    # elif config.arch in torchvision_models.__dict__.keys():
    #     student = torchvision_models.__dict__[config.arch]()
    #     teacher = torchvision_models.__dict__[config.arch]()
    #     embed_dim = student.fc.weight.shape[1]
        
    elif config.arch in svtr.__dict__.keys():
        student = svtr.__dict__[config.arch](drop_path_rate=config.drop_path_rate)
        teacher = svtr.__dict__[config.arch]()
        embed_dim = student.embed_dim[-1]
    else:
        print(f"Unknow architecture: {config.arch}")
        
    # import pdb
    # pdb.set_trace()
    # multi-crop wrapper handles forward with inputs of different resolutions
    seg_type = config.model_seg_type
    if seg_type == 'SegHead':
        seg_type = segmentor.__dict__[seg_type](in_channels=config.model_seg_channel, mla_channels=128, mlahead_channels=64, num_classes=2)
    else:
        seg_type = segmentor.__dict__[seg_type](in_channels=config.model_seg_channel)
        
    student = ABIDINOModel(
        student,
        seg_type,
        DINOHead(embed_dim, config.out_dim, use_bn=config.use_bn_in_head, norm_last_layer=config.norm_last_layer, ))
    
    teacher = ABIDINOModel(
        teacher,
        None,
        DINOHead(embed_dim, config.out_dim, config.use_bn_in_head), )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[config.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
        
    student = nn.parallel.DistributedDataParallel(student, device_ids=[config.gpu], find_unused_parameters=True)
    # teacher and student start with the same weights
    # teacher_without_ddp.load_state_dict(student.module.state_dict())
    teacher_without_ddp.backbone.load_state_dict(student.module.backbone.state_dict())
    teacher_without_ddp.head.load_state_dict(student.module.head.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
        
    print(f"Student and Teacher are built: they are both {config.arch} network.")
    
    return student, teacher,teacher_without_ddp

