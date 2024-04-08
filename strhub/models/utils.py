from pathlib import PurePath, Path
from typing import Sequence

import torch
from torch import nn
from collections import OrderedDict
from omegaconf import ListConfig, OmegaConf
import yaml


class InvalidModelError(RuntimeError):
    """Exception raised for any model-related error (creation, loading)"""


_WEIGHTS_URL = {
    'parseq-tiny': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt',
    'parseq-patch16-224': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_small_patch16_224-fcf06f5a.pt',
    'parseq': {
        'url': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
        'local': 'pretrained/parseq-bb5792a6.pt'
    },
    'parseq_custom': {
        'url': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
        'local': 'pretrained/parseq-bb5792a6.pt'
    },
    'abinet': 'https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt',
    'trba': 'https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt',
    'vitstr': 'https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt',
    'crnn': 'https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt',
}


def _get_config(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parents[2]
    with open(root / 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / f'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    # Workaround for now: manually cast the lr to the correct type.
    config['lr'] = float(config['lr'])
    return config


def _get_model_class(key):
    if 'parseq_custom' in key:
        from .parseq_custom.system import PARSeq as ModelClass
    elif 'parseq_viptr' in key:
        from .parseq_viptr.system import PARSeq as ModelClass
    else:
        raise InvalidModelError(f"Unable to find model class for '{key}'")
    return ModelClass


def get_pretrained_weights(experiment, model:nn.Module=None):
    if isinstance(experiment, ListConfig):
        experiment=OmegaConf.to_object(experiment)
        state_dict = model.state_dict()
        for ckpt_path in experiment:
            if Path(ckpt_path).exists():
                ckpt=torch.load(ckpt_path, map_location='cpu')
                # load teacher's backbone state_dict into encoder of model
                if ckpt.get('teacher', None) is not None:
                    for name, param in ckpt['teacher'].items():
                        if name.startswith('module'):
                            name = name[7:]
                            if name.startswith('backbone'):
                                if state_dict.get('encoder.'+name[9:], None) is not None:
                                    state_dict['encoder.'+name[9:]] = param
                elif ckpt.get('state_dict', None) is not None:
                    for name, param in ckpt['state_dict'].items():
                        if name.startswith('decoder'):
                            state_dict[name]=param
        return state_dict
    else:
        if Path(experiment).exists() and Path(experiment).is_file():
            ckpt=torch.load(experiment, map_location='cpu')
            # load teacher's backbone state_dict into encoder of model
            if ckpt.get('teacher', None) is not None:
                state_dict = model.state_dict()
                for name, param in ckpt['teacher'].items():
                    if name.startswith('module'):
                        name = name[7:]
                        if name.startswith('backbone'):
                            if state_dict.get('encoder.'+name[9:], None) is not None:
                                state_dict['encoder.'+name[9:]] = param
                return state_dict
            elif ckpt.get('state_dict', None) is not None:
                state_dict = model.state_dict()
                for name, param in ckpt['state_dict'].items():
                    state_dict[name]=param
                return state_dict


def create_model(experiment: str, pretrained: bool = False, **kwargs):
    try:
        config = _get_config(experiment, **kwargs)
    except FileNotFoundError:
        raise InvalidModelError(f"No configuration found for '{experiment}'") from None
    ModelClass = _get_model_class(experiment)
    model = ModelClass(**config)
    if pretrained:
        model.load_state_dict(get_pretrained_weights(experiment))
    return model


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    if checkpoint_path.startswith('pretrained='):
        model_id = checkpoint_path.split('=', maxsplit=1)[1]
        model = create_model(model_id, True, **kwargs)
    else:
        ModelClass = _get_model_class(checkpoint_path)
        model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
