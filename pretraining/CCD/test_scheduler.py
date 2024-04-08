from Dino.modules import utils
from src.option import _parse_arguments
import logging

config = _parse_arguments()
logging.info(config)
from src.dataset import _get_databaunch
train_dataloader = _get_databaunch(config)
config.iter_num = len(train_dataloader)
logging.info(f"each epoch iteration: {config.iter_num}")
config.epochs = int(config.training_epochs * len(train_dataloader) * (
                config.batch_size_per_gpu * utils.get_world_size()) / config.imgnet_based) + 1
    
print(f'training epochs is {config.epochs}')
print('config info')
    
for attr in ['lr', 'batch_size_per_gpu', 'min_lr', 'training_epochs', 'warmup_epoch', 'imgnet_based']:
    print(attr, getattr(config, attr))

lr_schedule = utils.cosine_iter_scheduler(
    config.lr * (config.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
    config.min_lr,
    config.training_epochs * len(train_dataloader),
    warmup_iters=int(
        (config.warmup_epoch * config.imgnet_based) / (config.batch_size_per_gpu * utils.get_world_size())),
)