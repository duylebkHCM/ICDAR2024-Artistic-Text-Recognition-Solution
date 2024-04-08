import os
import torch
from Dino.utils.utils import Config, Logger, MyConcatDataset
from Dino.utils.util import Averager
from Dino.dataset.datasetsupervised_kmeans import ImageDatasetSelfSupervisedKmeans
from Dino.dataset.dataset import collate_fn_filter_none
from fastai.vision import *
import torch.distributed as dist

def _get_databaunch(config):
    def _get_dataset(ds_type, paths, is_training, config, **kwargs):
        kwargs.update({
            'img_h': config.dataset_image_height,
            'img_w': config.dataset_image_width,
            'max_length': config.dataset_max_length,
            'case_sensitive': config.dataset_case_sensitive,
            'data_aug': config.dataset_data_aug,
            'deteriorate_ratio': config.dataset_deteriorate_ratio,
            'multiscales': config.dataset_multiscales,
            'data_portion': config.dataset_portion,
            'filter_single_punctuation': config.dataset_filter_single_punctuation,
            'mask': config.dataset_mask
        })
        datasets = []
        for p in paths:
            subfolders = [f.path for f in os.scandir(p) if f.is_dir()]
            if subfolders:  # Concat all subfolders
                datasets.append(_get_dataset(ds_type, subfolders, is_training, config, **kwargs))
            else:
                datasets.append(ds_type(path=p, is_training=is_training, **kwargs))
        if len(datasets) > 1:
            return MyConcatDataset(datasets)
        else:
            return datasets[0]

    bunch_kwargs = {}
    ds_kwargs = {}
    bunch_kwargs['collate_fn'] = collate_fn_filter_none
    dataset_class = ImageDatasetSelfSupervisedKmeans
    if config.dataset_augmentation_severity is not None:
        ds_kwargs['augmentation_severity'] = config.dataset_augmentation_severity
    ds_kwargs['supervised_flag'] = ifnone(config.model_contrastive_supervised_flag, False)
    train_ds = _get_dataset(dataset_class, config.dataset_train_roots, True, config, **ds_kwargs)
    if dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=True)
    else:
        sampler = torch.util.data.RandomSampler(train_ds)
        
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        sampler=sampler,
        batch_size=config.batch_size_per_gpu,
        num_workers=config.dataset_num_workers,
        collate_fn=collate_fn_filter_none,
        pin_memory=config.dataset_pin_memory,
        drop_last=True,
    )
    return train_dataloader
