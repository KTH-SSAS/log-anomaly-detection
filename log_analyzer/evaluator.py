import torch
import numpy as np


class Evaluator():
    def __init__(self, trainer):
        """Creates an Evaluator instance that wraps around a Trainer instance to provide additional evaluation """
        self.trainer = trainer
        self.data = [] # List? Dict? Tensor?
        # What data needs to be stored?
        # For each line:
        # GT,
        # model output,
        # predicted tokens (easily computed from model output),
        # Loss for the line (can also be computed as needed from GT+model output)
        # Red flag (i.e. readteam event or not)
        # Day + second

    def compute_evaluation_data(self, datafile):
        """Computes the data required for evaluation for the given data file"""

    def plot_losses_by_line(self):
        """Computes and plots the 75/95/99 percentiles of anomaly score (loss) for each second"""

    def get_auc_score(self):
        """Computes AUC score"""

    def plot_ROC_curve(self):
        """Plots the ROC (Receiver Operator Characteristic), i.e. TP-FP tradeoff, curve"""