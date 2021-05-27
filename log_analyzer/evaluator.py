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
        if (
            not caching
            or self.plot_losses_by_line_cache is None
            or self.plot_losses_by_line_cache[2] != percentiles
        ):
            smoothing = min(
                max(1, int(smoothing)), len(self.data["losses"])
            )  # Ensure smoothing is an int and in the range [1, len(losses)]
            if not smoothing % 2:
                smoothing += 1  # Ensure smoothing is odd

            losses_by_second = []
            seconds = []

            # Create a list of losses for each second
            for second, loss in tqdm(zip(self.data["seconds"], self.data["losses"])):
                if second not in seconds:
                    losses_by_second.append(np.array([loss]))
                    seconds.append(second)
                else:
                    idx = seconds.index(second)
                    losses_by_second[idx] = np.concatenate(
                        (losses_by_second[idx], [loss])
                    )
            if caching:
                self.plot_losses_by_line_cache = (
                    seconds,
                    losses_by_second,
                    percentiles,
                )

        if caching:
            seconds, losses_by_second, percentiles = self.plot_losses_by_line_cache

        plotting_data = []
        for p in percentiles:
            # iterate over losses_by_second (which contains the list of losses at each second)
            # and apply np.percentile to compute the "p"th percentile (e.g. 75th percentile)
            # loss at that second, then append this list of "p"th percentiles to plotting_data
            percentile_data = map(lambda x: np.percentile(x, p), losses_by_second)
            # map returns a generator, so apply list() to generate the list
            percentile_data = list(percentile_data)
            # apply the desired smoothing
            smoothed_data = (
                np.convolve(percentile_data, np.ones(smoothing), "valid") / smoothing
            )
            # To avoid boundary effects, 'valid' is used in the convolution (thus producing a list
            # that is shorter than the input), and the first and last values are re-inserted
            # to restore the original length
            for _ in range(int((smoothing - 1) / 2)):
                smoothed_data = np.insert(smoothed_data, 0, percentile_data[0])
                smoothed_data = np.insert(
                    smoothed_data, len(smoothed_data), percentile_data[-1]
                )
            plotting_data.append(smoothed_data)

        # Extract all red team events
        red_seconds = self.data["seconds"][self.data["red_flags"] != 0]
        red_losses = self.data["losses"][self.data["red_flags"] != 0]

        # Extract the top X outlier non-red team events
        # X is 1 per minute
        outlier_count = len(seconds) // 60
        blue_losses = self.data["losses"][self.data["red_flags"] == 0]
        blue_seconds = self.data["seconds"][self.data["red_flags"] == 0]
        # Negate the list so we can pick the highest values (i.e. the lowest -ve values)
        outlier_indices = np.argpartition(-blue_losses, outlier_count)[:outlier_count]
        blue_losses = blue_losses[outlier_indices]
        blue_seconds = blue_seconds[outlier_indices]
        print(len(blue_losses))

        # plot the percentile ranges
        for idx in range(len(plotting_data) - 1):
            plt.fill_between(
                seconds, plotting_data[idx], plotting_data[idx + 1], color=colors[idx]
            )
        # plot the redteam events
        plt.plot(red_seconds, red_losses, "r+", label="Red team events")
        # plot the non-redteam outliers
        plt.plot(blue_seconds, blue_losses, "bo", label="Outlier normal events")

        plt.xlabel("x - Time (seconds)")
        plt.ylabel(f"y - Percentiles of loss {tuple(percentiles)}")
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
            label=f"ROC curve (area = {auc_score}",
        )
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlabel("x - False Positive Rate")
        plt.ylabel("y - True Positive Rate")
        plt.title("Receiver Operating Characteristic curve")
        return auc_score
