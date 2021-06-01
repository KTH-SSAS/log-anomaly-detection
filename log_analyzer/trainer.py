from log_analyzer.config.model_config import LSTMConfig
from log_analyzer.config.trainer_config import TrainerConfig
import torch
import torch.nn as nn
from log_analyzer.model.lstm import Fwd_LSTM, Bid_LSTM, LogModel
import log_analyzer.model.auxiliary as auxiliary
from log_analyzer.evaluator import Evaluator
from abc import ABC, abstractmethod

# TODO name this something more descriptive, it might be used as a wrapper around both transformer/LSTM
class Trainer(ABC):

    @property
    @abstractmethod
    def model(self) -> LogModel:
        pass

    def __init__(self, config : TrainerConfig, verbose, checkpoint_dir):

        self.config = config
        
        # Check GPU
        self.cuda = torch.cuda.is_available()

        self.jagged = config.jagged
        self.bidirectional = config.bidirectional
        self.checkpoint_dir = checkpoint_dir

        if self.cuda:
            self.model.cuda()

        # Create settings for training.
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        self.early_stopping = auxiliary.EarlyStopping(
            patience=config.early_stop_patience, verbose=verbose, path=checkpoint_dir)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        # Create evaluator
        self.evaluator = Evaluator()

    def compute_loss(self, output, Y, lengths, mask):
        """Computes the loss for the given model output and ground truth."""
        if self.jagged:
            if self.bidirectional:
                targets = Y[:, 1 : max(lengths) - 1]
                token_losses = self.criterion(
                    output.transpose(1, 2), targets
                )
                masked_losses = token_losses * mask[:, 1 : max(lengths) - 1]
            else:
                targets = Y[:, :max(lengths)]
                token_losses = self.criterion(
                    output.transpose(1, 2), targets
                )
                masked_losses = token_losses * mask[:, : max(lengths)]
            line_losses = torch.sum(masked_losses, dim=1)
        else:
            targets = Y
            token_losses = self.criterion(output.transpose(1, 2), Y)
            line_losses = torch.mean(token_losses, dim=1)
        loss = torch.mean(line_losses, dim=0)

        # Return the loss, as well as extra details like loss per line
        return loss, line_losses, targets

    def optimizer_step(self, loss):
        """Performs one step of optimization on the given loss."""
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.early_stopping(loss, self.model)

    def split_batch(self, batch):
        """Splits a batch into variables containing relevant data."""
        X = batch['x']
        Y = batch['t']
        if self.jagged:
            L = batch['length']
            M = batch['mask']
        else:
            L = None
            M = None
        if self.cuda:
            X = X.cuda()
            Y = Y.cuda()
            if self.jagged:
                L = L.cuda()
                M = M.cuda()

        return X, Y, L, M

    def train_step(self, batch):
        """Defines a single training step. Feeds data through the model, computes the loss and makes an optimization step."""

        self.model.train()
        self.optimizer.zero_grad()

        # Split the batch into input, ground truth, etc.
        X, Y, L, M = self.split_batch(batch)

        # Apply the model to input to produce the output
        output, *_ = self.model(X, lengths=L)

        # Compute the loss for the output
        loss, *_ = self.compute_loss(output, Y, lengths=L, mask=M)

        # Take an optimization step based on the loss
        self.optimizer_step(loss)

        return loss, self.early_stopping.early_stop

    def eval_step(self, batch, store_eval_data=False):
        """Defines a single evaluation step. Feeds data through the model and computes the loss."""
        # TODO add more metrics, like perplexity.
        self.model.eval()

        # Split the batch into input, ground truth, etc.
        X, Y, L, M = self.split_batch(batch)

        # Apply the model to input to produce the output
        output, *_ = self.model(X, lengths=L)

        # Compute the loss for the output
        loss, line_losses, targets = self.compute_loss(output, Y, lengths=L, mask=M)

        # Save the results if desired
        if store_eval_data:
            preds = torch.argmax(output, dim=-1)
            self.evaluator.add_evaluation_data(
                targets, preds, line_losses, batch["second"], batch["red"],
            )

        # Return both the loss and the output token probabilities
        return loss, output


class LSTMTrainer(Trainer):
    """Trainer class for forward and bidirectional LSTM model"""
    @property
    def model(self):
        if self.lstm is None:
            raise RuntimeError("Model not intialized!")
        return self.lstm

    def __init__(self, config : TrainerConfig, lstm_config : LSTMConfig, checkpoint_dir, verbose):

        model = Bid_LSTM if config.bidirectional else Fwd_LSTM
        # Create a model
        self.lstm = model(lstm_config)

        super().__init__(config, verbose, checkpoint_dir)
