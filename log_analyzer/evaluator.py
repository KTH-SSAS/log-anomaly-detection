import torch
import numpy as np
import matplotlib.pyplot as plt



#TODO: support for tiered model
class Evaluator:
    def __init__(self):
        """Creates an Evaluator instance that provides methods for model evaluation"""
        self.reset_evaluation_data()
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
        self.data["log_lines"].extend(log_line.detach())
        self.data["predictions"].extend(predictions.detach())
        self.data["losses"] = np.concatenate((self.data["losses"], losses.detach()))
        self.data["seconds"] = np.concatenate((self.data["seconds"], seconds.detach()))
        self.data["days"] = np.concatenate((self.data["days"], days.detach()))
        self.data["red_flags"] = np.concatenate(
            (self.data["red_flags"], red_flags.detach())
        )

    def reset_evaluation_data(self):
        """Delete the stored evaluation data"""
        self.data = {
            "log_lines": [],
            "predictions": [],
            "losses": np.array([]),
            "seconds": np.array([], int),
            "days": np.array([], int),
            "red_flags": np.array([], bool),
        }

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
        # Compute the average loss
        average_loss = np.average(self.data["losses"])
        # Assuming the loss is cross entropy loss, the perplexity is the exponential of the loss
        perplexity = np.exp(average_loss)
        return perplexity

    def get_auc_score(self):
        """Computes AUC score (area under the ROC curve)"""

    def plot_losses_by_line(self, percentiles=[75, 95, 99], smoothing=1):
        """Computes and plots the given (default 75/95/99) percentiles of anomaly score
        (loss) by line for each second"""
        # TODO: support for data spanning more than one day (the data["days"] entry is currently ignored)
        smoothing = min(
            max(1, int(smoothing)), len(self.data["losses"])
        )  # Ensure smoothing is an int and in the range [1, len(losses)]
        if not smoothing % 2:
            smoothing += 1  # Ensure smoothing is odd

        losses_by_second = []
        seconds = []

        # Create a list of losses for each second
        for second, loss in zip(self.data["seconds"], self.data["losses"]):
            second = second
            loss = loss
            if second not in seconds:
                losses_by_second.append(np.array([loss]))
                seconds.append(second)
            else:
                idx = seconds.index(second)
                losses_by_second[idx] = np.concatenate((losses_by_second[idx], [loss]))

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

        # plot the percentile ranges
        for idx in range(len(plotting_data) - 1):
            plt.fill_between(seconds, plotting_data[idx], plotting_data[idx + 1])
        # plot the redteam events
        plt.plot(red_seconds, red_losses, "r+")

        plt.xlabel("x - Time (seconds)")
        plt.ylabel(f"y - Percentiles of loss {tuple(percentiles)}")
        plt.title("Aggregate line losses by time")

    def plot_ROC_curve(self):
        """Plots the ROC (Receiver Operator Characteristic) curve, i.e. TP-FP tradeoff"""
