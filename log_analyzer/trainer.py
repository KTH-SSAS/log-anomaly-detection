from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import log_analyzer.model.early_stopping as early_stopping
from log_analyzer.application import Application
from log_analyzer.config.model_config import LSTMConfig, TransformerConfig
from log_analyzer.config.trainer_config import TrainerConfig
from log_analyzer.evaluator import Evaluator
from log_analyzer.model.lstm import Bid_LSTM, Fwd_LSTM, LogModel
from log_analyzer.model.transformer import Transformer

# TODO name this something more descriptive, it might be used as a wrapper
# around both transformer/LSTM


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
        self.criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
        self.early_stopping = early_stopping.EarlyStopping(patience=config.early_stop_patience, path=checkpoint_dir)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
        self.use_scheduler = bool(config.scheduler_step_size)
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Create evaluator
        self.evaluator = Evaluator()

    def compute_loss(self, output: torch.Tensor, Y, lengths, mask: torch.Tensor):
        """Computes the loss for the given model output and ground truth."""
        targets = Y
        if lengths is not None:
            if self.model.bidirectional:
                token_losses = self.criterion(output.transpose(1, 2), targets)
                masked_losses = token_losses * mask
            else:
                token_losses = self.criterion(output.transpose(1, 2), targets)
                masked_losses = token_losses * mask
            line_losses = torch.sum(masked_losses, dim=1)
        else:
            token_losses = self.criterion(output.transpose(1, 2), Y)
            line_losses = torch.mean(token_losses, dim=1)
        loss = torch.mean(line_losses, dim=0)

        # Return the loss, as well as extra details like loss per line
        return loss, line_losses, targets

    def optimizer_step(self, loss: torch.Tensor):
        """Performs one step of optimization on the given loss."""
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        if self.use_scheduler:
            self.scheduler.step()
        if self.config.early_stopping:
            self.early_stopping(loss, self.model)

    def split_batch(self, batch: dict):
        """Splits a batch into variables containing relevant data."""
        X = batch["input"]
        Y = batch["target"]

        # Optional fields
        L = batch.get("length")
        M = batch.get("mask")

        if self.cuda:
            X = X.cuda()
            Y = Y.cuda()
            if M is not None:
                M = M.cuda()

        return X, Y, L, M

    def train_step(self, batch):
        """Defines a single training step.

        Feeds data through the model, computes the loss and makes an
        optimization step.
        """

        self.model.train()
        self.optimizer.zero_grad()

        # Split the batch into input, ground truth, etc.
        X, Y, L, M = self.split_batch(batch)

        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                # Apply the model to input to produce the output
                output, *_ = self.model(X, lengths=L, mask=M)

                # Compute the loss for the output
                loss, *_ = self.compute_loss(output, Y, lengths=L, mask=M)
        else:
            # Apply the model to input to produce the output
            output, *_ = self.model(X, lengths=L, mask=M)

            # Compute the loss for the output
            loss, *_ = self.compute_loss(output, Y, lengths=L, mask=M)

        # Take an optimization step based on the loss
        self.optimizer_step(loss)

        return loss, self.early_stopping.early_stop

    def eval_step(self, batch, store_eval_data=False):
        """Defines a single evaluation step.

        Feeds data through the model and computes the loss.
        """
        # TODO add more metrics, like perplexity.
        self.model.eval()

        # Split the batch into input, ground truth, etc.
        X, Y, L, M = self.split_batch(batch)

        # Apply the model to input to produce the output
        output, *_ = self.model(X, lengths=L, mask=M)

        # Compute the loss for the output
        loss, line_losses, targets = self.compute_loss(output, Y, lengths=L, mask=M)

        # Save the results if desired
        if store_eval_data:
            preds = torch.argmax(output, dim=-1)
            self.evaluator.add_evaluation_data(
                targets,
                preds,
                batch["user"],
                line_losses,
                batch["second"],
                batch["red"],
            )
            self.evaluator.test_loss += loss
            self.evaluator.test_count += 1

        # Return both the loss and the output token probabilities
        return loss, output


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

        model = Bid_LSTM if bidirectional else Fwd_LSTM
        # Create a model
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
        # Create a model
        self.transformer = Transformer(transformer_config)

        super().__init__(config, checkpoint_dir)
