import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics


# TODO: support for tiered model
class Evaluator:
    def __init__(self):
        """Creates an Evaluator instance that provides methods for model evaluation"""
        self.reset_evaluation_data(reset_caches=True)

    def add_evaluation_data(
        self, log_line, predictions, losses, seconds, days, red_flags
    ):
        """Extend the data stored in self.data with the inputs"""
        self.data["log_lines"] = np.concatenate(
            (self.data["log_lines"], log_line.cpu().detach().flatten())
        )
        self.data["predictions"] = np.concatenate(
            (self.data["predictions"], predictions.cpu().detach().flatten())
        )
        self.data["losses"] = np.concatenate(
            (self.data["losses"], losses.cpu().detach())
        )
        self.data["seconds"] = np.concatenate(
            (self.data["seconds"], seconds.cpu().detach())
        )
        self.data["days"] = np.concatenate((self.data["days"], days.cpu().detach()))
        self.data["red_flags"] = np.concatenate(
            (self.data["red_flags"], red_flags.cpu().detach())
        )
        # Reset the caches whenever new data is added
        self.reset_caches()

    def reset_evaluation_data(self, reset_caches=True):
        """Delete the stored evaluation data"""
        self.data = {
            "log_lines": np.array([], int),
            "predictions": np.array([], int),
            "losses": np.array([]),
            "seconds": np.array([], int),
            "days": np.array([], int),
            "red_flags": np.array([], bool),
        }
        if reset_caches:
            self.reset_caches()

    def reset_caches(self):
        """Resets all the caches"""
        self.plot_losses_by_line_cache = None

    def get_metrics(self):
        """Computes and returns all metrics"""
        metrics = [
            self.get_token_accuracy(),
            self.get_token_perplexity(),
            self.get_auc_score(),
        ]
        return metrics

    def get_token_accuracy(self):
        """Computes the accuracy of the model token prediction"""
        return metrics.accuracy_score(self.data["log_lines"], self.data["predictions"])

    def get_token_perplexity(self):
        """Computes and returns the perplexity of the model token prediction"""
        # Compute the average loss
        average_loss = np.average(self.data["losses"])
        # Assuming the loss is cross entropy loss, the perplexity is the exponential of the loss
        perplexity = np.exp(average_loss)
        return perplexity

    def get_auc_score(self, fp_rate=None, tp_rate=None):
        """Computes AUC score (area under the ROC curve)"""
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
        caching=False,
    ):
        """Computes and plots the given (default 75/95/99) percentiles of anomaly score
        (loss) by line for each second"""
        # TODO: support for data spanning more than one day (the data["days"] entry is currently ignored)
        smoothing = min(
            max(1, int(smoothing)), len(self.data["losses"])
        )  # Ensure smoothing is an int and in the range [1, len(losses)]
        if not smoothing % 2:
            smoothing += 1  # Ensure smoothing is odd

        plotting_data = [[] for _ in percentiles]
        # Create a list of losses for each second
        seconds = np.unique(self.data["seconds"])
        for second in tqdm(seconds):
            losses = self.data["losses"][self.data["seconds"] == second]
            for idx, p in enumerate(percentiles):
                percentile_data = np.percentile(losses, p)
                plotting_data[idx].append(percentile_data)

        # apply the desired smoothing
        for idx, _ in enumerate(plotting_data):
            smoothed_data = (
                np.convolve(plotting_data[idx], np.ones(smoothing), "same") / smoothing
            )
            # Adjust the first and last (smoothing-1)/2 entries to avoid boundary effects
            for i in range(int((smoothing-1)/2)):
                # The first and last few entries are only averaged over (smoothing+1)/2 + i entries
                smoothed_data[i] *= smoothing/((smoothing+1)/2 + i)
                smoothed_data[-(i+1)] *= smoothing/((smoothing+1)/2 + i)
            plotting_data[idx] = smoothed_data

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
                seconds, plotting_data[idx], plotting_data[idx + 1], color=colors[idx]
            )
        # plot the non-redteam outliers
        plt.plot(blue_seconds, blue_losses, "bo", label="Outlier normal events")
        # plot the redteam events
        plt.plot(red_seconds, red_losses, "r+", label="Red team events")

        plt.xlabel("Time (seconds)")
        plt.ylabel(f"Percentiles of loss {tuple(percentiles)}")
        plt.title("Aggregate line losses by time")

    def plot_ROC_curve(self):
        """Plots the ROC (Receiver Operating Characteristic) curve, i.e. TP-FP tradeoff
        Also returns the corresponding auc score"""
        fp_rate, tp_rate, _ = metrics.roc_curve(
            self.data["red_flags"], self.data["losses"], pos_label=1
        )
        auc_score = self.get_auc_score()
        plt.plot(
            fp_rate,
            tp_rate,
            color="orange",
            lw=2,
            label=f"ROC curve (area = {auc_score:.2f})",
        )
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic curve")
        plt.legend()
        return auc_score
