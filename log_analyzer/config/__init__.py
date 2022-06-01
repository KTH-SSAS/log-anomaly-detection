from dataclasses import dataclass
from typing import List

from log_analyzer.config.config import Config


@dataclass
class TrainerConfig(Config):
    train_files: List[str]
    validation_files: List[str]
    test_files: List[str]
    shuffle_train_data: bool
    train_batch_size: int
    eval_batch_size: int
    epochs: int
    learning_rate: float
    early_stopping: bool
    early_stop_patience: int
    scheduler_gamma: float
    scheduler_step_size: int
    mixed_precision: bool
    gradient_accumulation: int
    validations_per_epoch: int
    gradient_clip: float
    warmup_period: int
    per_epoch_lr_reduction: float
