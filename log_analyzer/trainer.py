from pathlib import Path

import torch
from torch.cuda.amp.grad_scaler import GradScaler

from log_analyzer.application import Application
from log_analyzer.config import TrainerConfig
from log_analyzer.model import early_stopping
from log_analyzer.model.lstm import LogModel


def calculate_gradient_norm(model: torch.nn.Module):
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm: torch.Tensor = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]),
        2,
    )
    return total_norm.item()


class Trainer:
    def __init__(self, config: TrainerConfig, model: LogModel, checkpoint_dir: Path):

        self.config = config
        # Ensure gradient_accumulation is at least 1
        self.config.gradient_accumulation = max(self.config.gradient_accumulation, 1)

        self.model = model

        self.accumulated_steps = 0

        # Check GPU
        self.using_cuda = Application.instance().using_cuda

        self.checkpoint_dir: Path = checkpoint_dir

        if self.using_cuda:
            self.model.cuda()

        # Create settings for training.
        self.earlystopping = early_stopping.EarlyStopping(patience=config.early_stop_patience, path=checkpoint_dir)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma,
        )
        self.use_scheduler = bool(config.scheduler_step_size)
        if self.config.mixed_precision:
            self.scaler: GradScaler = torch.cuda.amp.GradScaler()

    def early_stopping(self, val_loss):
        """Performs early stopping check after validation, if enabled."""
        if self.config.early_stopping:
            self.earlystopping(val_loss, self.model)

    def optimizer_step(self, loss: torch.Tensor):
        """Performs one step of optimization on the given loss."""
        using_mp = self.config.mixed_precision
        if using_mp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        gradient_norm = calculate_gradient_norm(self.model)

        self.accumulated_steps = (self.accumulated_steps + 1) % self.config.gradient_accumulation
        if self.accumulated_steps == 0:
            if using_mp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        if self.use_scheduler:
            self.scheduler.step()

        return gradient_norm

    def load_model_weights(self, file_pointer):
        state_dict = torch.load(file_pointer)
        self.model.load_state_dict(state_dict)

    def train_step(self, split_batch, validation=False):
        """Defines a single training step.

        Feeds data through the model, computes the loss and makes an
        optimization step.

        split_batch: should contain X, Y, L, M
            X: input
            Y: target
            L: sequence lengths
            M: sequence masks

        validation: if set to True no backprop will be performed
        """
        X = split_batch["X"]
        Y = split_batch["Y"]
        L = split_batch["L"]
        M = split_batch["M"]

        if not validation:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                # Apply the model to input to produce the output, provide targets to receive loss
                _, loss = self.model(X, lengths=L, mask=M, targets=Y)
        else:
            # Apply the model to input to produce the output, provide targets to receive loss
            _, loss = self.model(X, lengths=L, mask=M, targets=Y)

        # Take an optimization step based on the loss
        gradient_norm = self.optimizer_step(loss) if not validation else 0.0

        return loss, gradient_norm, self.earlystopping.early_stop
