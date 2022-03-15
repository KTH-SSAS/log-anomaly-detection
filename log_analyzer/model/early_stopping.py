import copy
import logging
from pathlib import Path

import numpy as np
import torch

from log_analyzer.application import TRAINER_LOGGER


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a
    given patience."""

    def __init__(self, patience=4, delta=0, path: Path = Path("./")):
        """
        Args:
            patience (int): How long (batches) to wait after last time validation loss improved.
                            Default: 4
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: './'
        """
        self.patience = patience
        self.patience_counter = 0
        self.call_counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.saved_val_loss = np.Inf
        self.delta = delta
        self.path = path / "checkpoint.pt"
        self.model_state_dict = None
        self.logger = logging.getLogger(TRAINER_LOGGER)

    def __call__(self, val_loss, model):
        if val_loss < self.val_loss_min - self.delta:
            self.store_state_dict(val_loss, model)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            self.logger.debug(
                "EarlyStopping counter: %d out of %d. Best loss: %f",
                self.patience_counter,
                self.patience,
                self.val_loss_min,
            )
            if self.patience_counter >= self.patience:
                self.early_stop = True
        # Save checkpoint if best model isn't already saved, at most every 2nd time early_stopping is called
        if self.call_counter % 2 == 0 and self.val_loss_min < self.saved_val_loss:
            self.save_checkpoint()
        self.call_counter += 1

    def store_state_dict(self, val_loss, model):
        """Stores the model dict of the best performing model so far."""
        self.model_state_dict = copy.deepcopy(model.state_dict())

        self.logger.debug("Loss decreased (%.6f --> %.6f).", self.val_loss_min, val_loss)
        self.val_loss_min = val_loss

    def save_checkpoint(self):
        """Saves model to file if the model has improved since last time it was
        saved."""
        self.logger.info("Best Loss: %.6f, Saving model ...", self.val_loss_min)
        self.saved_val_loss = self.val_loss_min
        torch.save(self.model_state_dict, self.path)
