import torch
import numpy as np



class Evaluator:
    def __init__(self):
        """Creates an Evaluator instance that provides methods for model evaluation"""
        self.data = {
            "log_lines": [],
            "predictions": [],
            "losses": [],
            "seconds": [],
            "days": [],
            "red_flags": [],
        }
        # What data needs to be stored?
        # For each line:
        # predicted tokens
        # Loss for the line

        # Can be streamed from the data reader on demand?
        # Ground Truth,
        # Red flag (i.e. redteam event or not)
        # Day + second

    def add_evaluation_data(
        self, log_line, predictions, losses, seconds, days, red_flags
    ):
        """Extend the data stored in self.data with the inputs"""
        self.data["log_lines"].extend(log_line)
        self.data["predictions"].extend(predictions)
        self.data["losses"].extend(losses)
        self.data["seconds"].extend(seconds)
        self.data["days"].extend(days)
        self.data["red_flags"].extend(red_flags)

    def reset_evaluation_data(self):
        """Delete the stored evaluation data"""
        for key in self.data.keys():
            self.data[key] = []

    def get_metrics(self):
        """Computes and returns all metrics"""

    def get_token_accuracy(self):
        """Computes the accuracy of the model token prediction"""
        # Flatten the log_line and predictions lists
        flattened_lines = np.array([])
        for line in self.data["log_lines"]:
            flattened_lines = np.concatenate((flattened_lines, line.numpy()))
        flattened_preds = np.array([])
        for pred in self.data["predictions"]:
            flattened_preds = np.concatenate((flattened_preds, pred.numpy()))

        # Subtract each predicted token from the actual token
        # Thus a value of 0 in matches indicates a correct prediction
        matches = flattened_lines - flattened_preds

        # % accuracy = 1 - number_of_non_matches/number_of_tokens
        accuracy = 1 - (np.count_nonzero(matches) / len(matches))
        return accuracy

    def get_token_perplexity(self):
        """Computes and returns the perplexity of the model token prediction"""

    def get_auc_score(self):
        """Computes AUC score (area under the ROC curve)"""

    def plot_losses_by_line(self):
        """Computes and plots the 75/95/99 percentiles of anomaly score (loss) for each second"""

    def plot_ROC_curve(self):
        """Plots the ROC (Receiver Operator Characteristic) curve, i.e. TP-FP tradeoff"""
