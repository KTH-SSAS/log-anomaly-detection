import torch
import numpy as np



class Evaluator:
    def __init__(self):
        """Creates an Evaluator instance that provides methods for model evaluation"""
        self.data = {
            "predictions": [],
            "losses": [],
            "red_flags": [],
            "days": [],
            "seconds": [],
        }
        # What data needs to be stored?
        # For each line:
        # predicted tokens
        # Loss for the line

        # Can be streamed from the data reader on demand?
        # Ground Truth,
        # Red flag (i.e. redteam event or not)
        # Day + second

    def get_evaluation_data(self, model, datafile):
        """Computes the data required for evaluation for the given data file"""

    def reset_evaluation_data(self):
        """Delete the stored evaluation data"""
        self.data = {
            "predictions": [],
            "losses": [],
            "red_flags": [],
            "days": [],
            "seconds": [],
        }

    def get_metrics(self):
        """Computes and returns all metrics"""

    def get_token_accuracy(self):
        """Computes the accuracy of the model token prediction"""

    def get_token_perplexity(self):
        """Computes and returns the perplexity of the model token prediction"""

    def get_auc_score(self):
        """Computes AUC score (area under the ROC curve)"""

    def plot_losses_by_line(self):
        """Computes and plots the 75/95/99 percentiles of anomaly score (loss) for each second"""

    def plot_ROC_curve(self):
        """Plots the ROC (Receiver Operator Characteristic) curve, i.e. TP-FP tradeoff"""
