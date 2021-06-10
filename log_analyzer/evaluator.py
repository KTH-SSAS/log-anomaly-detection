import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics


class Evaluator:
    def __init__(self):
        """Creates an Evaluator instance that provides methods for model evaluation"""
        self.data_is_trimmed = False
        self.data_is_normalized = False
        self.reset_evaluation_data()

    def add_evaluation_data(
        self, log_line, predictions, users, losses, seconds, red_flags
    ):
        """Extend the data stored in self.data with the inputs"""
        log_line = log_line.cpu().detach().flatten()
        predictions = predictions.cpu().detach().flatten()
        losses = losses.cpu().detach()
        seconds = seconds.cpu().detach()
        red_flags = red_flags.cpu().detach()
        # Check that there's enough space left for all the entries
        if len(self.data["losses"]) < self.index["losses"] + len(log_line):
            self.data["users"] = np.concatenate(
                (self.data["users"], np.zeros(1050000, float))
            )
            self.data["losses"] = np.concatenate(
                (self.data["losses"], np.zeros(1050000, float))
            )
            self.data["seconds"] = np.concatenate(
                (self.data["seconds"], np.zeros(1050000, int))
            )
            self.data["red_flags"] = np.concatenate(
                (self.data["red_flags"], np.zeros(1050000, bool))
            )

        for key, new_data in zip(
            ["users", "losses", "seconds", "red_flags"],
            [users, losses, seconds, red_flags],
        ):
            self.data[key][self.index[key] : self.index[key] + len(new_data)] = new_data
            self.index[key] += len(new_data)

        # Compute the token accuracy for this batch
        batch_token_accuracy = metrics.accuracy_score(log_line, predictions)
        new_token_count = self.token_count + len(log_line)
        new_token_accuracy = (
            self.token_accuracy * self.token_count
            + batch_token_accuracy * len(log_line)
        ) / new_token_count
        self.token_count = new_token_count
        self.token_accuracy = new_token_accuracy

    def reset_evaluation_data(self):
        """Delete the stored evaluation data"""
        self.data = {
            "users": np.zeros(0, int),
            "losses": np.zeros(0, float),
            "seconds": np.zeros(0, int),
            "red_flags": np.zeros(0, bool),
        }
        self.index = {
            "users": 0,
            "losses": 0,
            "seconds": 0,
            "red_flags": 0,
        }
        self.token_accuracy = 0
        self.token_count = 0
        self.data_is_trimmed = False

    def trim_evaluation_data(self):
        """Trims any remaining allocated entries for the evaluation data lists"""
        for key in self.data.keys():
            self.data[key] = self.data[key][: self.index[key]]
        self.data_is_trimmed = True

    def normalize_losses(self):
        """Performs user-level anomaly score normalization by subtracting the average
        anomaly score of the user from each event (log line).
        Mainly relevant to word tokenization"""
        # Do not re-normalize data
        if self.data_is_normalized:
            return
        # Loop over every user
        average_losses = np.ones_like(self.data["losses"])
        for user in tqdm(np.unique(self.data["users"])):
            user_indices = self.data["users"] == user
            # Compute the average loss for this user
            average_loss = np.average(self.data["losses"][user_indices])
            average_losses[user_indices] = average_loss
        # Apply the normalization
        self.data["losses"] *= average_losses
        self.data_is_normalized = True

    def get_metrics(self):
        """Computes and returns all metrics"""
        metrics = {
            "token_accuracy": self.get_token_accuracy(),
            "token_perplexity": self.get_token_perplexity(),
            "auc_score": self.get_auc_score(),
        }
        return metrics

    def get_token_accuracy(self):
        """Returns the accuracy of the model token prediction"""
        return self.token_accuracy

    def get_token_perplexity(self):
        """Computes and returns the perplexity of the model token prediction"""
        if not self.data_is_trimmed:
            self.trim_evaluation_data()
        # Compute the average loss
        average_loss = np.average(self.data["losses"])
        # Assuming the loss is cross entropy loss, the perplexity is the exponential of the loss
        perplexity = np.exp(average_loss)
        return perplexity

    def get_auc_score(self, fp_rate=None, tp_rate=None):
        """Computes AUC score (area under the ROC curve)"""
        if not self.data_is_trimmed:
            self.trim_evaluation_data()
        # Compute fp and tp rates if not supplied
        if fp_rate == None or tp_rate == None:
            fp_rate, tp_rate, _ = metrics.roc_curve(
                self.data["red_flags"], self.data["losses"], pos_label=1
            )
        auc_score = metrics.auc(fp_rate, tp_rate)
        return auc_score

    def plot_losses_by_line(
        self,
        percentiles=[75, 95, 99],
        smoothing=1,
        colors=["darkorange", "gold"],
        ylim=(-1, -1),
    ):
        """Computes and plots the given (default 75/95/99) percentiles of anomaly score
        (loss) by line for each second. Smoothing indicates how many seconds are processed
        as one batch for percentile calculations (e.g. 60 means percentiles are computed
        for every minute)."""
        if not self.data_is_trimmed:
            self.trim_evaluation_data()

        plotting_data = [[] for _ in percentiles]
        # Create a list of losses for each segment
        seconds = np.unique(self.data["seconds"])
        segments = [seconds[i] for i in range(0, len(seconds), smoothing)]
        for idx in tqdm(range(len(segments))):
            segment_start = np.searchsorted(self.data["seconds"], segments[idx])
            if idx == len(segments) - 1:
                segment_end = len(self.data["losses"])
            else:
                segment_end = np.searchsorted(self.data["seconds"], segments[idx + 1])
            segment_losses = self.data["losses"][segment_start:segment_end]
            for perc_idx, p in enumerate(percentiles):
                percentile_data = np.percentile(segment_losses, p)
                plotting_data[perc_idx].append(percentile_data)

        # Extract all red team events
        red_seconds = self.data["seconds"][self.data["red_flags"] != 0]
        red_losses = self.data["losses"][self.data["red_flags"] != 0]

        # Extract the top X (1 per minute of data) outlier non-red team events
        outlier_count = len(seconds) // 60
        blue_losses = self.data["losses"][self.data["red_flags"] == 0]
        blue_seconds = self.data["seconds"][self.data["red_flags"] == 0]
        # Negate the list so we can pick the highest values (i.e. the lowest -ve values)
        outlier_indices = np.argpartition(-blue_losses, outlier_count)[:outlier_count]
        blue_losses = blue_losses[outlier_indices]
        blue_seconds = blue_seconds[outlier_indices]

        # plot the percentile ranges
        for idx in range(len(plotting_data) - 1):
            plt.fill_between(
                segments, plotting_data[idx], plotting_data[idx + 1], color=colors[idx]
            )
        # plot the non-redteam outliers
        plt.plot(blue_seconds, blue_losses, "bo", label="Outlier normal events")
        # plot the redteam events
        plt.plot(red_seconds, red_losses, "r+", label="Red team events")
        if ylim[0] >= 0 and ylim[1] > 0:
            plt.ylim(ylim)
        plt.xlabel("Time (seconds)")
        plt.ylabel(f"Loss, {tuple(percentiles)} percentiles")
        plt.title("Aggregate line losses by time")

    def plot_ROC_curve(self, xaxis="FPR"):
        """Plots the ROC (Receiver Operating Characteristic) curve, i.e. TP-FP tradeoff
        Also returns the corresponding auc score. Options for xaxis are:
        'FPR': False-positive rate. The default.
        'alerts': # of alerts per second (average) the FPR would be equivalent to.
        'alerts-FPR': What % of produced alerts would be false alerts."""
        if not self.data_is_trimmed:
            self.trim_evaluation_data()
        fp_rate, tp_rate, _ = metrics.roc_curve(
            self.data["red_flags"], self.data["losses"], pos_label=1
        )
        auc_score = self.get_auc_score()
        red_flag_count = sum(self.data["red_flags"])
        non_red_flag_count = len(self.data["red_flags"]) - red_flag_count
        xlabel = "False Positive Rate"
        if xaxis.lower() == "alerts":
            # Multiply the fp_rate by the number of events in the eval set to convert
            # fp_rate into fp's per second
            fp_rate *= red_flag_count
            xlabel += " - Alerts per second"
        elif xaxis.lower() == "alerts-fpr":
            total_alerts = fp_rate * non_red_flag_count + tp_rate * red_flag_count
            fp_rate = tp_rate * red_flag_count / total_alerts
            xlabel += " - Precision"
        plt.plot(
            fp_rate,
            tp_rate,
            color="orange",
            lw=2,
            label=f"ROC curve (area = {auc_score:.2f})",
        )
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlabel(xlabel)
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic curve")
        plt.legend()
        return auc_score
