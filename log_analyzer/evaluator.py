import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

import wandb
from log_analyzer.application import Application
from log_analyzer.model.lstm import LogModel, LSTMLanguageModel
from log_analyzer.tokenizer.tokenizer import Char_tokenizer


def create_attention_matrix(
    model: LSTMLanguageModel,
    sequences,
    output_dir,
    lengths=None,
    mask=None,
    token_map_file=None,
):
    """Plot attention matrix over batched input.

    Will produce one matrix plot for each entry in batch, in the
    designated output directory. For word level tokenization, the
    function will also produce an matrix for the avergae attention
    weights in the batch.
    """
    if model.attention is None:
        raise RuntimeError("Can not create an attention matrix for a model without attention!")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    skip_sos = not model.bidirectional
    tokenization = "word" if lengths is None else "char"

    _, lstm_out, _ = model.forward(sequences, lengths=lengths, mask=mask)
    _, attention_matrix_batch = model.attention(lstm_out, mask)

    def set_ticks():
        word_tick_labels = [
            "<sos>",
            "src user",
            "src domain",
            "dest user",
            "dest domain",
            "src PC",
            "dest PC",
            "auth type",
            "login type",
            "auth orient",
            "success/fail",
            "<eos>",
        ]
        input_labels = word_tick_labels[1:-1] if skip_sos else word_tick_labels[:-1]
        attention_labels = word_tick_labels[1:-1] if skip_sos else word_tick_labels[:-1]
        label_labels = word_tick_labels[2:] if skip_sos else word_tick_labels[1:]
        ax.set_xlabel("Positions attended over")
        ax.set_xticks(range(matrix.shape[0]))
        ax.set_xticklabels(attention_labels, rotation="45")
        ax.set_yticks(range(matrix.shape[1]))
        ax.set_yticklabels(input_labels)
        ax.set_ylabel("Input token")
        twin.set_yticks(range(matrix.shape[1]))
        twin.set_yticklabels(label_labels)
        twin.set_ylabel("Predicted label")

    if lengths is None:
        # Batch average of attention over positions
        matrix = attention_matrix_batch.mean(dim=0)
        _, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(matrix.detach().numpy())
        twin = ax.twinx()
        twin.matshow(matrix.detach().numpy())
        set_ticks()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model.attention.attention_type}_batchAverage.png"))

    _, ax = plt.subplots(figsize=(10, 10))
    for i, matrix in enumerate(attention_matrix_batch):
        seq = sequences[i]

        if lengths is not None:
            matrix = matrix[: lengths[i] - 1, : lengths[i] - 1]

        ax.matshow(matrix.detach().numpy())
        if lengths is not None:
            string = Char_tokenizer.detokenize_line(seq[: lengths[i] - 1])
            ax.set_xticks(range(len(string)))
            ax.set_xticklabels(string, fontsize="small")
            ax.set_yticks(range(len(string)))
            ax.set_yticklabels(string, fontsize="small")
        else:
            twin = ax.twinx()
            twin.matshow(matrix.detach().numpy())
            set_ticks()
            plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"{model.attention.attention_type}_{tokenization}_#{i}.png"))
        plt.cla()


class Evaluator:
    def __init__(self, model: LogModel):
        """Creates an Evaluator instance that provides methods for model
        evaluation."""
        self.model = model
        self.data_is_prepared = False
        self.reset_evaluation_data()
        self.use_wandb = Application.instance().wandb_initialized

    # TEMP TODO: Still WIP, complete move of eval_step from trainer classes to Evaluator
    @torch.no_grad()
    def eval_step(self, split_batch, store_eval_data=False):
        """Defines a single evaluation step.

        Feeds data through the model and computes the loss.

        split_batch: should contain X, Y, L, M
            X: input
            Y: target
            L: sequence lengths
            M: sequence masks
        """
        X = split_batch["X"]
        Y = split_batch["Y"]
        L = split_batch["L"]
        M = split_batch["M"]

        users = split_batch["user"]
        seconds = split_batch["second"]
        red_flags = split_batch["red_flag"]

        self.model.eval()

        # Apply the model to input to produce the output
        output, *_ = self.model(X, lengths=L, mask=M)

        # Compute the loss for the output
        loss, line_losses = self.model.compute_loss(output, Y, lengths=L, mask=M)

        # Save the results if desired
        if store_eval_data:
            preds = torch.argmax(output, dim=-1)
            self.add_evaluation_data(
                Y,
                preds,
                users,
                line_losses,
                seconds,
                red_flags,
            )
            self.test_loss += loss
            self.test_count += 1

        # Return both the loss and the output token probabilities
        return loss, output

    def run_all(self):
        r"""Performs standard evaluation on the model. Assumes the model has been trained
        and the evaluator has been populated with evaluation data (see eval_step)"""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        # Get generic metrics
        evaluator_metrics = self.get_metrics()

        # get line losses plot
        self.plot_line_loss_percentiles(
            percentiles=[75, 95, 99], smoothing=300, ylim=(-1, -1), outliers=1, legend=False
        )
        if self.use_wandb:
            wandb.log({"Aggregate line losses": wandb.Image(plt)})
        plt.clf()

        # get roc curve
        _, roc_plot = self.plot_roc_curve()
        if self.use_wandb:
            wandb.log({"ROC Curve": roc_plot})

        # get pr curve
        AP_score, pr_plot = self.plot_pr_curve()
        if self.use_wandb:
            wandb.log({"PR Curve": pr_plot})
        evaluator_metrics["eval/AP"] = AP_score

        # get normalised line losses plot
        self.plot_line_loss_percentiles(
            percentiles=[75, 95, 99], smoothing=300, ylim=(-1, -1), outliers=1, legend=False, normalised=True
        )
        if self.use_wandb:
            wandb.log({"Aggregate line losses (normalised)": wandb.Image(plt)})
        plt.clf()

        # get normalised roc curve
        evaluator_metrics["eval/AUC_(normalised)"], roc_plot = self.plot_roc_curve(
            title="ROC (normalised)", normalised=True
        )
        if self.use_wandb:
            wandb.log({"ROC Curve (normalised)": roc_plot})

        # get normalised pr curve
        evaluator_metrics["eval/AP_(normalised)"], pr_plot = self.plot_pr_curve(
            title="PR Curve (normalised)", normalised=True
        )
        if self.use_wandb:
            wandb.log({"PR Curve": pr_plot})

        # Log the evaluation results
        if self.use_wandb and wandb.run is not None:
            for key in evaluator_metrics:
                wandb.run.summary[key] = evaluator_metrics[key]
        return evaluator_metrics

    def add_evaluation_data(self, log_line, predictions, users, losses, seconds, red_flags):
        """Extend the data stored in self.data with the inputs."""
        log_line = log_line.cpu().detach().flatten()
        predictions = predictions.cpu().detach().flatten()
        losses = losses.cpu().detach()
        seconds = seconds.cpu().detach()
        red_flags = red_flags.cpu().detach()
        # Check that there's enough space left for all the entries
        if len(self.data["losses"]) < self.index["losses"] + len(log_line):
            # Adding entries 1'050'000 at a time provides a nice balance of efficiency and memory usage.
            # Most days have just over 7 million log lines, so incrementing with 1'000'000 is inefficient
            self.data["users"] = np.concatenate((self.data["users"], np.zeros(1050000, float)))
            self.data["losses"] = np.concatenate((self.data["losses"], np.zeros(1050000, float)))
            self.data["seconds"] = np.concatenate((self.data["seconds"], np.zeros(1050000, int)))
            self.data["red_flags"] = np.concatenate((self.data["red_flags"], np.zeros(1050000, bool)))

        for key, new_data in zip(
            ["users", "losses", "seconds", "red_flags"],
            [users, losses, seconds, red_flags],
        ):
            self.data[key][self.index[key] : self.index[key] + len(new_data)] = new_data
            self.index[key] += len(new_data)

        # Update the metatag, i.e. data is prepared and normalised data is ready
        self.data_is_prepared = False
        # Update token accuracy including this batch
        batch_token_accuracy = metrics.accuracy_score(log_line, predictions)
        new_token_count = self.token_count + len(log_line)
        new_token_accuracy = (
            self.token_accuracy * self.token_count + batch_token_accuracy * len(log_line)
        ) / new_token_count
        self.token_count = new_token_count
        self.token_accuracy = new_token_accuracy

    def reset_evaluation_data(self):
        """Delete the stored evaluation data."""
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
        self.test_loss = 0
        self.test_count = 0
        self.data_is_prepared = False

    def prepare_evaluation_data(self):
        """Prepares the evaluation data by:

        1. Trimming any remaining allocated entries for the evaluation data lists
        2. Sorting the data (by second) if it is not sorted
        """
        for key in self.data:
            # Ignore normalised_losses
            if key == "normalised_losses":
                continue
            self.data[key] = self.data[key][: self.index[key]]
        # Check if the data is sorted
        if not np.all(np.diff(self.data["seconds"]) >= 0):
            # Sort the data by seconds
            sorted_indices = np.argsort(self.data["seconds"])
            for key in ["users", "losses", "seconds", "red_flags"]:
                self.data[key] = self.data[key][sorted_indices]
        # Compute final test loss
        self.test_loss /= max(self.test_count, 1)
        self.test_count = 1
        # Prepared the normalised losses
        self._normalise_losses()

        self.data_is_prepared = True

    def _normalise_losses(self):
        """Performs user-level anomaly score normalization by subtracting the
        average anomaly score of the user from each event (log line).

        Mainly relevant to word tokenization
        """
        # Loop over every user
        average_losses = np.ones_like(self.data["losses"])
        for user in tqdm(np.unique(self.data["users"])):
            user_indices = self.data["users"] == user
            # Compute the average loss for this user
            average_loss = np.average(self.data["losses"][user_indices])
            average_losses[user_indices] = average_loss
        # Apply the normalization
        self.data["normalised_losses"] = self.data["losses"] - average_losses

    def get_metrics(self):
        """Computes and returns all metrics."""
        metrics = {
            "eval/loss": self.get_test_loss(),
            "eval/token_accuracy": self.get_token_accuracy(),
            "eval/token_perplexity": self.get_token_perplexity(),
            "eval/AUC": self.get_auc_score(),
            "eval/AP": self.get_ap_score(),
        }
        return metrics

    def get_test_loss(self):
        """Returns the accuracy of the model token prediction."""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        return float(self.test_loss)

    def get_token_accuracy(self):
        """Returns the accuracy of the model token prediction."""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        return self.token_accuracy

    def get_token_perplexity(self):
        """Computes and returns the perplexity of the model token
        prediction."""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        # Compute the average loss
        average_loss = np.average(self.data["losses"])
        # Assuming the loss is cross entropy loss, the perplexity is the
        # exponential of the loss
        perplexity = np.exp(average_loss)
        return perplexity

    def get_auc_score(self, fp_rate=None, tp_rate=None, normalised=False):
        """Computes AUC score (area under the ROC curve)"""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        # Compute fp and tp rates if not supplied
        if fp_rate is None or tp_rate is None:
            # Get the relevant data - normalised or not
            losses = self.data["normalised_losses"] if normalised else self.data["losses"]
            fp_rate, tp_rate, _ = metrics.roc_curve(self.data["red_flags"], losses, pos_label=1)
        auc_score = metrics.auc(fp_rate, tp_rate)
        return auc_score

    def plot_line_loss_percentiles(
        self,
        percentiles=[75, 95, 99],
        smoothing=1,
        colors=["darkorange", "gold"],
        ylim=(-1, -1),
        outliers=10,
        legend=True,
        normalised=False,
    ):
        """Computes and plots the given (default 75/95/99) percentiles of
        anomaly score (loss) by line for each segment.

        Smoothing indicates how many seconds are processed as one batch
        for percentile calculations (e.g. 60 means percentiles are
        computed for every minute). Outliers determines how many non-
        red team outliers are plotted onto the graph (per hour of data).
        """
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        # Ensure percentiles is sorted in ascending order
        percentiles = sorted(percentiles)
        # Ensure smoothing > 0, and int
        smoothing = int(smoothing)
        if smoothing <= 0:
            smoothing = 1

        # Get the relevant data - normalised or not
        losses = self.data["normalised_losses"] if normalised else self.data["losses"]

        plotting_data = [[] for _ in percentiles]
        # Create a list of losses for each segment
        seconds = np.unique(self.data["seconds"])
        segments = [seconds[i] for i in range(0, len(seconds), smoothing)]
        for idx in tqdm(range(len(segments))):
            segment_start = np.searchsorted(self.data["seconds"], segments[idx])
            if idx == len(segments) - 1:
                segment_end = len(losses)
            else:
                segment_end = np.searchsorted(self.data["seconds"], segments[idx + 1])
            segment_losses = losses[segment_start:segment_end]
            for perc_idx, p in enumerate(percentiles):
                percentile_data = np.percentile(segment_losses, p)
                plotting_data[perc_idx].append(percentile_data)

        # Extract all red team events
        red_seconds = self.data["seconds"][self.data["red_flags"] != 0]
        red_losses = losses[self.data["red_flags"] != 0]

        if outliers > 0:
            # Extract the top X ('outliers' per hour of data) outlier non-red
            # team events
            outlier_count = int(len(seconds) * outliers // 3600)
            blue_losses = losses[self.data["red_flags"] == 0]
            blue_seconds = self.data["seconds"][self.data["red_flags"] == 0]
            # Negate the list so we can pick the highest values (i.e. the
            # lowest -ve values)
            outlier_indices = np.argpartition(-blue_losses, outlier_count)[:outlier_count]
            blue_losses = blue_losses[outlier_indices]
            blue_seconds = blue_seconds[outlier_indices]
            blue_seconds = blue_seconds / (3600 * 24)  # convert to days

        # plot the percentile ranges
        # Convert x-axis to days
        red_seconds = red_seconds / (3600 * 24)
        segments = [s / (3600 * 24) for s in segments]
        for idx in range(len(plotting_data) - 2, -1, -1):
            plt.fill_between(
                segments,
                plotting_data[idx],
                plotting_data[idx + 1],
                color=colors[idx],
                label=f"{percentiles[idx]}-{percentiles[idx + 1]} Percentile",
            )
        # plot the non-red-team outliers
        if outliers > 0:
            plt.plot(blue_seconds, blue_losses, "bo", label="Outlier normal events")
        # plot the red team events
        plt.plot(red_seconds, red_losses, "r+", label="Red team events")
        if ylim[0] >= 0 and ylim[1] > 0:
            plt.ylim(ylim)
        plt.xlabel("Time (day)")
        plt.ylabel(f"Loss, {tuple(percentiles)} percentiles")
        if legend:
            plt.legend()
        plt.title("Aggregate line losses by time")

    def plot_roc_curve(self, title="ROC", normalised=False):
        """Plots the ROC (Receiver Operating Characteristic) curve, i.e. TP-FP
        tradeoff. Also returns the corresponding auc score.

        Options for xaxis are:
        'FPR': False-positive rate. The default.
        'alerts': # of alerts per second (average) the FPR would be equivalent to.
        'alerts-FPR': What % of produced alerts would be false alerts.
        """
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        auc_score = self.get_auc_score()

        # Get the relevant data - normalised or not
        losses = self.data["normalised_losses"] if normalised else self.data["losses"]

        full_fp_rate, full_tp_rate, _ = metrics.roc_curve(self.data["red_flags"], losses, pos_label=1)
        # Scale fp_rate, tp_rate down to contain <10'000 values
        # E.g. if original length is 1'000'000, only take every 100th value
        step_size = (len(full_fp_rate) // 10000) + 1
        fp_rate = full_fp_rate[::step_size]
        tp_rate = full_tp_rate[::step_size]
        # Ensure the last value in full_fp_rate and full_tp_rate is included
        if fp_rate[-1] != full_fp_rate[-1]:
            fp_rate = np.append(fp_rate, full_fp_rate[-1])
            tp_rate = np.append(tp_rate, full_tp_rate[-1])
        # Erase the full fp and tp lists
        full_fp_rate = full_tp_rate = []
        if self.use_wandb:
            # ROC Curve is to be uploaded to wandb, so plot using a "fixed"
            # version of their plot.roc_curve function
            table = wandb.Table(
                columns=["class", "fpr", "tpr"],
                data=list(zip(["" for _ in fp_rate], fp_rate, tp_rate)),
            )
            wandb_plot = wandb.plot_table(
                "wandb/area-under-curve/v0",
                table,
                {"x": "fpr", "y": "tpr", "class": "class"},
                {
                    "title": title,
                    "x-axis-title": "False positive rate",
                    "y-axis-title": "True positive rate",
                },
            )
            return auc_score, wandb_plot
        else:
            # Plot using scikit-learn and matplotlib
            red_flag_count = sum(self.data["red_flags"])
            non_red_flag_count = len(self.data["red_flags"]) - red_flag_count
            xlabel = "False Positive Rate"

            plt.plot(
                fp_rate,
                tp_rate,
                color="orange",
                lw=2,
                label=f"ROC curve (area = {auc_score:.2f})",
            )
            plt.xlabel(xlabel)
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend()
            return auc_score, plt

    def get_ap_score(self, normalised=False):
        """Computes AP score (average precision)"""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        # Get the relevant data - normalised or not
        losses = self.data["normalised_losses"] if normalised else self.data["losses"]
        ap_score = metrics.average_precision_score(self.data["red_flags"], losses)
        return ap_score

    def plot_pr_curve(self, title="Precision-Recall Curve", normalised=False):
        """Plots the Precision-Recall curve, and returns the corresponding auc
        score."""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()

        # Get the relevant data - normalised or not
        losses = self.data["normalised_losses"] if normalised else self.data["losses"]

        full_precision, full_recall, _ = metrics.precision_recall_curve(self.data["red_flags"], losses, pos_label=1)
        # Get average precision score as a summary score for PR
        AP_score = metrics.average_precision_score(self.data["red_flags"], losses)

        # Scale precision, recall down to contain <10'000 values
        # E.g. if original length is 1'000'000, only take every 100th value
        step_size = (len(full_precision) // 10000) + 1
        precision = full_precision[::step_size]
        recall = full_recall[::step_size]

        # Ensure the last value in full_precision and full_recall is included
        if precision[-1] != full_precision[-1]:
            precision = np.append(precision, full_precision[-1])
            recall = np.append(recall, full_recall[-1])
        # Erase the full fp and tp lists
        full_precision = full_recall = full_thresh = []

        # Round to 5 digits
        precision = list(map(lambda x: round(x, 5), precision))
        recall = list(map(lambda x: round(x, 5), recall))

        if self.use_wandb:
            # PR Curve is to be uploaded to wandb, so plot using a "fixed"
            # version of their plot.pr_curve function
            table = wandb.Table(
                columns=["class", "recall", "precision"],
                data=list(zip(["" for _ in recall], recall, precision)),
            )
            wandb_plot = wandb.plot_table(
                "wandb/area-under-curve/v0",
                table,
                {"x": "recall", "y": "precision", "class": "class"},
                {
                    "title": "Precision v. Recall",
                    "x-axis-title": "Recall",
                    "y-axis-title": "Precision",
                },
            )
            return AP_score, wandb_plot
        else:
            # Plot using scikit-learn and matplotlib
            xlabel = "Recall"
            ylabel = "Precision"
            plt.plot(
                recall,
                precision,
                color="orange",
                lw=2,
                label=f"Intrusion events",
            )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            return AP_score, plt
