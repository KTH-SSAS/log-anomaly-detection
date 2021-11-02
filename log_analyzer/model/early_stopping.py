import logging
import os

import numpy as np
import torch

from log_analyzer.application import TRAINER_LOGGER


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, path='./'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(path, 'checkpoint.pt')
        self.model_state_dict = None

    def __call__(self, val_loss, model):

        score = -val_loss

        logger = logging.getLogger(TRAINER_LOGGER)

        if self.best_score is None:
            self.best_score = score
            self.save_state_dict(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter % 10 == 0:
                logger.debug('EarlyStopping counter: %d out of %d. Best loss: %f',
                             self.counter, self.patience, -self.best_score)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_state_dict(val_loss, model)
            self.counter = 0

    def save_state_dict(self, val_loss, model):
        self.model_state_dict = model.state_dict()

        logging.getLogger(TRAINER_LOGGER).debug(
            'Loss decreased (%.6f --> %.6f).', self.val_loss_min, val_loss)
        self.val_loss_min = val_loss

    def save_checkpoint(self):
        '''Saves model when validation loss decrease.'''

        logging.getLogger(TRAINER_LOGGER).info(
            'Best Loss: %.6f, Saving model ...', self.val_loss_min)
        torch.save(self.model_state_dict, self.path)
