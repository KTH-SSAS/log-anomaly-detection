import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# TODO: support for tiered model
class Evaluator:
    def __init__(self):
        """Creates an Evaluator instance that provides methods for model evaluation"""
        self.reset_evaluation_data(reset_caches=True)

    def add_evaluation_data(
        self, log_line, predictions, losses, seconds, days, red_flags
    ):
        """Extend the data stored in self.data with the inputs"""
        self.data["log_lines"].extend(log_line.cpu().detach())
        self.data["predictions"].extend(predictions.cpu().detach())
        self.data["losses"] = np.concatenate((self.data["losses"], losses.cpu().detach()))
        self.data["seconds"] = np.concatenate((self.data["seconds"], seconds.cpu().detach()))
        self.data["days"] = np.concatenate((self.data["days"], days.cpu().detach()))
        self.data["red_flags"] = np.concatenate(
            (self.data["red_flags"], red_flags.cpu().detach())
        )
        # Reset the caches whenever new data is added
        self.reset_caches()

    def reset_evaluation_data(self, reset_caches=True):
        """Delete the stored evaluation data"""
        self.data = {
            "log_lines": [],
            "predictions": [],
            "losses": np.array([]),
            "seconds": np.array([], int),
            "days": np.array([], int),
            "red_flags": np.array([], bool),
        }
        if reset_caches:
            self.reset_caches()
    
    def reset_caches(self):
        """Resets all the caches"""
        self.get_token_accuracy_cache = None
        self.plot_losses_by_line_cache = None

    def get_metrics(self):
        """Computes and returns all metrics"""
        metrics = [
            self.get_token_accuracy(),
            self.get_token_perplexity(),
            self.get_auc_score(),
        ]
        return metrics

    def get_token_accuracy(self, caching=True):
        """Computes the accuracy of the model token prediction"""
        # Flatten the log_line and predictions lists
        if not caching or self.get_token_accuracy_cache is None:
            matches = 0
            tokens = 0
            for line_num, line in enumerate(tqdm(self.data["log_lines"])):
                for token_num, token in enumerate(line):
                    matches += (token == self.data["predictions"][line_num][token_num])
                    tokens += 1

            # % accuracy = 1 - number_of_non_matches/number_of_tokens
            accuracy = float(matches) / tokens
            if caching:
                self.get_token_accuracy_cache = accuracy
        else:
            accuracy = self.get_token_accuracy_cache
        return accuracy

    def get_token_perplexity(self):
        """Computes and returns the perplexity of the model token prediction"""
        # Compute the average loss
        average_loss = np.average(self.data["losses"])
        # Assuming the loss is cross entropy loss, the perplexity is the exponential of the loss
        perplexity = np.exp(average_loss)
        return perplexity

    # TODO: Implement once access to redteam data has been implemented
    def get_auc_score(self):
        """Computes AUC score (area under the ROC curve)"""
        raise NotImplementedError()

    def plot_losses_by_line(self, percentiles=[75, 95, 99], smoothing=1, colors=["darkorange", "gold"], caching=False):
        """Computes and plots the given (default 75/95/99) percentiles of anomaly score
        (loss) by line for each second"""
        # TODO: support for data spanning more than one day (the data["days"] entry is currently ignored)
        if not caching or self.plot_losses_by_line_cache is None or self.plot_losses_by_line_cache[2] != percentiles:
            smoothing = min(
                max(1, int(smoothing)), len(self.data["losses"])
            )  # Ensure smoothing is an int and in the range [1, len(losses)]
            if not smoothing % 2:
                smoothing += 1  # Ensure smoothing is odd

            losses_by_second = []
            seconds = []

            # Create a list of losses for each second
            for second, loss in tqdm(zip(self.data["seconds"], self.data["losses"])):
                second = second
                loss = loss
                if second not in seconds:
                    losses_by_second.append(np.array([loss]))
                    seconds.append(second)
                else:
                    idx = seconds.index(second)
                    losses_by_second[idx] = np.concatenate((losses_by_second[idx], [loss]))
            if caching:
                self.plot_losses_by_line_cache = (seconds, losses_by_second, percentiles)

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

        # Extract outlier non-red team events
        blue_losses = self.data["losses"][self.data["red_flags"] == 0]
        blue_seconds = self.data["seconds"][self.data["red_flags"] == 0]
        indices = blue_losses > 10
        blue_losses = blue_losses[indices]
        blue_seconds = blue_seconds[indices]

        # plot the percentile ranges
        for idx in range(len(plotting_data) - 1):
            plt.fill_between(seconds, plotting_data[idx], plotting_data[idx + 1], color=colors[idx])
        # plot the redteam events
        plt.plot(red_seconds, red_losses, "r+", label="Red team events")
        # plot the non-redteam outliers
        plt.plot(blue_seconds, blue_losses, "bo", label="Outlier normal events")

        plt.xlabel("x - Time (seconds)")
        plt.ylabel(f"y - Percentiles of loss {tuple(percentiles)}")
        plt.title("Aggregate line losses by time")

    # TODO: Implement once access to redteam data has been implemented
    def plot_ROC_curve(self):
        """Plots the ROC (Receiver Operator Characteristic) curve, i.e. TP-FP tradeoff"""
        raise NotImplementedError()
