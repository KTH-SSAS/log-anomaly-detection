from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.cuda.amp.grad_scaler import GradScaler

import log_analyzer.model.early_stopping as early_stopping
from log_analyzer.application import Application
from log_analyzer.config.model_config import LSTMConfig, TieredLSTMConfig, TieredTransformerConfig, TransformerConfig
from log_analyzer.config.trainer_config import TrainerConfig
from log_analyzer.evaluator import Evaluator
from log_analyzer.model.lstm import BidLSTM, FwdLSTM, LogModel, TieredLSTM
from log_analyzer.model.transformer import TieredTransformer, Transformer


class Trainer(ABC):
    @property
    @abstractmethod
    def model(self) -> LogModel:
        pass

    def __init__(self, config: TrainerConfig, checkpoint_dir):

        self.config = config

        # Check GPU
        self.cuda = Application.instance().using_cuda

        self.checkpoint_dir = checkpoint_dir

        if self.cuda:
            self.model.cuda()

        # Create settings for training.
        self._EarlyStopping = early_stopping.EarlyStopping(patience=config.early_stop_patience, path=checkpoint_dir)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
        self.use_scheduler = bool(config.scheduler_step_size)
        self.scaler: Optional[GradScaler]
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Create evaluator
        self.evaluator = Evaluator(self.model)

    def early_stopping(self, val_loss):
        """Performs early stopping check after validation, if enabled."""
        if self.config.early_stopping:
            self._EarlyStopping(val_loss, self.model)

    def optimizer_step(self, loss: torch.Tensor):
        """Performs one step of optimization on the given loss."""
        if self.config.mixed_precision and isinstance(self.scaler, GradScaler):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        if self.use_scheduler:
            self.scheduler.step()

    def train_step(self, split_batch):
        """Defines a single training step.

        Feeds data through the model, computes the loss and makes an
        optimization step.

        split_batch: should contain X, Y, L, M
            X: input
            Y: target
            L: sequence lengths
            M: sequence masks
        """
        X = split_batch["X"]
        Y = split_batch["Y"]
        L = split_batch["L"]
        M = split_batch["M"]

        self.model.train()
        self.optimizer.zero_grad()

        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                # Apply the model to input to produce the output, provide targets to receive loss
                _, loss = self.model(X, lengths=L, mask=M, targets=Y)
        else:
            # Apply the model to input to produce the output, provide targets to receive loss
            _, loss = self.model(X, lengths=L, mask=M, targets=Y)

        # Take an optimization step based on the loss
        self.optimizer_step(loss)

        return loss, self._EarlyStopping.early_stop


class LSTMTrainer(Trainer):
    """Trainer class for forward and bidirectional LSTM model."""

    @property
    def model(self):
        if self.lstm is None:
            raise RuntimeError("Model not intialized!")
        return self.lstm

    def __init__(
        self,
        config: TrainerConfig,
        lstm_config: LSTMConfig,
        bidirectional,
        checkpoint_dir,
    ):

        model = BidLSTM if bidirectional else FwdLSTM
        # Create the model
        self.lstm = model(lstm_config)

        super().__init__(config, checkpoint_dir)


class TransformerTrainer(Trainer):
    """Trainer class for Transformer model."""

    @property
    def model(self):
        if self.transformer is None:
            raise RuntimeError("Model not initialized!")
        return self.transformer

    def __init__(
        self,
        config: TrainerConfig,
        transformer_config: TransformerConfig,
        checkpoint_dir,
    ):
        # Create the model
        self.transformer = Transformer(transformer_config)

        super().__init__(config, checkpoint_dir)


class TieredLSTMTrainer(Trainer):
    """Trainer class for tiered LSTM model."""

    @property
    def model(self):
        if self.lstm is None:
            raise RuntimeError("Model not intialized!")
        return self.lstm

    def __init__(
        self,
        config: TrainerConfig,
        lstm_config: TieredLSTMConfig,
        bidirectional,
        checkpoint_dir,
    ):
        # Create the model
        self.lstm = TieredLSTM(lstm_config, bidirectional)

        super().__init__(config, checkpoint_dir)


class TieredTransformerTrainer(Trainer):
    """Trainer class for tiered transformer model."""

    @property
    def model(self):
        if self.transformer is None:
            raise RuntimeError("Model not initialized!")
        return self.transformer

    def __init__(
        self,
        config: TrainerConfig,
        transformer_config: TieredTransformerConfig,
        bidirectional,
        checkpoint_dir,
    ):
        # Create the model
        self.transformer = TieredTransformer(transformer_config)
        super().__init__(config, checkpoint_dir)
