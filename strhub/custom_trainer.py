import logging
import math
import os
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union
from pytorch_lightning.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    LRSchedulerConfig,
    TRAIN_DATALOADERS,
)
from pytorch_lightning.strategies import (
    DDPFullyShardedNativeStrategy,
    DDPStrategy,
    ParallelStrategy,
    SingleDeviceStrategy,
    Strategy,
)
from pytorch_lightning.utilities import parsing
from pytorch_lightning.trainer.configuration_validator import verify_loop_configurations
from pytorch_lightning.loops.utilities import _parse_loop_limits
from pytorch_lightning.utilities.seed import isolate_rng

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)

class CustomTrainer(Trainer):
    def _run_train(self) -> None:
        self._pre_training_routine()

        with isolate_rng():
            self._run_sanity_check()

        # enable train mode
        assert self.model is not None
        self.model.train()
        torch.set_grad_enabled(True)

        self.fit_loop.trainer = self

        import pdb
        pdb.set_trace()
        
        with torch.autograd.set_detect_anomaly(self._detect_anomaly):
            self.fit_loop.run()