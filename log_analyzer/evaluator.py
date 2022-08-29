from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from torch import Tensor
from tqdm import tqdm

import wandb
from log_analyzer.application import Application
from log_analyzer.model.lstm import LogModel, LSTMLanguageModel, MultilineLogModel
from log_analyzer.tokenizer.tokenizer import CharTokenizer


def create_attention_matrix(
    model: LSTMLanguageModel,
    sequences,
    output_dir: Path,
    lengths=None,
    mask=None,
):
    """Plot attention matrix over batched input.

    Will produce one matrix plot for each entry in batch, in the
    designated output directory. For word level tokenization, the
    function will also produce an matrix for the avergae attention
    weights in the batch.
    """
    if model.attention is None:
        raise RuntimeError("Can not create an attention matrix for a model without attention!")

    if not output_dir.exists():
        output_dir.mkdir()

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
    plt.savefig(output_dir / f"{model.attention.attention_type}_batchAverage.png")

    _, ax = plt.subplots(figsize=(10, 10))
    for i, matrix in enumerate(attention_matrix_batch):
        seq = sequences[i]

        if lengths is not None:
            matrix = matrix[: lengths[i] - 1, : lengths[i] - 1]

        ax.matshow(matrix.detach().numpy())
        if lengths is not None:
            string = CharTokenizer.detokenize_line(seq[: lengths[i] - 1])
            ax.set_xticks(range(len(string)))
            ax.set_xticklabels(string, fontsize="small")
            ax.set_yticks(range(len(string)))
            ax.set_yticklabels(string, fontsize="small")
        else:
            twin = ax.twinx()
            twin.matshow(matrix.detach().numpy())
            set_ticks()
            plt.tight_layout()

        plt.savefig(output_dir / f"{model.attention.attention_type}_{tokenization}_#{i}.png")
        plt.cla()


class Evaluator:
    def __init__(self, model: LogModel, checkpoint_dir):
        """Creates an Evaluator instance that provides methods for model
        evaluation."""
        self.model = model
        self.data_is_prepared = False
        self.reset_evaluation_data()
        self.use_wandb = Application.instance().wandb_initialized
        self.checkpoint_dir = checkpoint_dir
        self.token_accuracy = torch.tensor(0, dtype=torch.float)
        self.token_count = torch.tensor(0, dtype=torch.long)
        self.eval_loss = torch.tensor(0, dtype=torch.float)
        self.eval_lines_count = torch.tensor(0, dtype=torch.long)
        self.skipped_line_count = torch.tensor(0, dtype=torch.long)
        if Application.instance().using_cuda:
            self.token_accuracy = self.token_accuracy.cuda()
            self.token_count = self.token_count.cuda()
            self.eval_loss = self.eval_loss.cuda()
            self.eval_lines_count = self.eval_lines_count.cuda()

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

        if M is not None:
            mask = M
        else:
            mask = None

        skipped_lines = None
        # Find which lines, if any, are "skipped lines" (i.e. ones without full context)
        # Check for any masked-out context lines - it's enough to check the first line of each sequence
        if isinstance(self.model, MultilineLogModel):
            skipped_lines = (mask[:, 0] == 0).unsqueeze(1).repeat(1, Y.shape[1])

        users = split_batch["user"]
        seconds = split_batch["second"]
        red_flags = split_batch["red_flag"]

        self.model.eval()

        # Apply the model to input to produce the output
        output, _ = self.model(X, lengths=L, mask=M)

        # Compute the loss for the output
        loss, line_losses = self.model.compute_loss(output, Y)

        # Save the results if desired
        if store_eval_data:
            if isinstance(self.model, MultilineLogModel) and not isinstance(
                self.model.criterion, torch.nn.CrossEntropyLoss
            ):
                # Multiline logmodels do not necessarily produce predictions over a discrete space that can/should be
                # argmaxed. If CrossEntropyLoss is not used, this means the predictions are placed in the continuous
                # sentence-embedding space. Therefore we cannot track token-accuracy and do not pass Y or preds to
                # add_evaluation_data().
                Y = None
                preds = None
            else:
                preds = torch.argmax(output, dim=-1)

            # Ensure the mask is the same shape as the rest of the tensors - for multiline models it might not be
            # If mask is larger, take the last X entries of mask where X is the shape of the other tensors
            if mask is not None:
                mask = mask[:, -users.shape[1] :]

            self.add_evaluation_data(
                users,
                seconds,
                red_flags,
                line_losses,
                log_line=Y,
                predictions=preds,
                mask=mask,
                skipped_lines=skipped_lines,
            )
            self.eval_loss += loss
            self.eval_lines_count += 1

        # Return both the loss and the output token probabilities
        return loss, output

    def add_evaluation_data(
        self,
        users: Tensor,
        seconds: Tensor,
        red_flags: Tensor,
        losses: Tensor,
        log_line: Optional[Tensor] = None,
        predictions: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        skipped_lines: Optional[Tensor] = None,
    ):
        """Extend the data stored in self.data with the inputs."""
        # Handle input from tiered models
        users = users.numpy().flatten()
        seconds = seconds.numpy().flatten()
        red_flags = red_flags.numpy().flatten()

        # Handle masked (i.e. padding) lines in the input
        if mask is not None:
            mask = mask.cpu().numpy().flatten()
            users = users[mask]
            seconds = seconds[mask]
            red_flags = red_flags[mask]

        losses = losses.cpu().flatten()
        if mask is not None:
            losses = losses[mask]

        # Check that there's enough space left for all the entries
        if len(self.data["losses"]) < self.index + len(users):
            # Adding entries 1'050'000 at a time provides a nice balance of efficiency and memory usage.
            # Most days have just over 7 million log lines, so incrementing with 1'000'000 is inefficient
            self.data["users"] = np.concatenate((self.data["users"], np.zeros(1050000, float)))
            self.data["losses"] = np.concatenate((self.data["losses"], np.zeros(1050000, float)))
            self.data["seconds"] = np.concatenate((self.data["seconds"], np.zeros(1050000, int)))
            self.data["red_flags"] = np.concatenate((self.data["red_flags"], np.zeros(1050000, bool)))
            self.data["skipped"] = np.concatenate((self.data["skipped"], np.zeros(1050000, bool)))

        if skipped_lines is not None:
            # trick to make mypy happy
            skipped_lines_dummy: Tensor = skipped_lines
            skipped_lines_dummy = skipped_lines_dummy.cpu().numpy().flatten()
            if mask is not None:
                skipped_lines_dummy = skipped_lines_dummy[mask]
            skipped_lines = skipped_lines_dummy
            # This flags lines that could not be evaluated by the model.
            # (e.g. multiline model, not full context available before this line)
            # Set skipped flag to 1 and loss to 0
            self.data["skipped"][self.index : self.index + len(users)] = skipped_lines

        for key, new_data in zip(
            ["users", "seconds", "red_flags"],
            [users, seconds, red_flags],
        ):
            self.data[key][self.index : self.index + len(new_data)] = new_data.squeeze()
        # Handle losses separately, since it might be None
        if losses is not None:
            self.data["losses"][self.index : self.index + len(losses)] = losses.squeeze()
        # Update the index
        self.index += len(users)

        # Update the metatag, i.e. data is prepared and normalised data is ready
        self.data_is_prepared = False
        # Update token accuracy including this batch
        if log_line is not None and predictions is not None:
            predictions = predictions.detach()
            if mask is not None:
                log_line = log_line.flatten(end_dim=1)[mask].flatten()
                predictions = predictions.flatten(end_dim=1)[mask].flatten()
            else:
                log_line = log_line.flatten()
                predictions = predictions.flatten()
            log_line_length = log_line.shape[0]
            batch_token_accuracy = torch.sum(log_line == predictions) / log_line_length
            new_token_count = self.token_count + log_line_length
            new_token_accuracy = (
                self.token_accuracy * self.token_count + batch_token_accuracy * log_line_length
            ) / new_token_count
            self.token_count = new_token_count
            self.token_accuracy = new_token_accuracy

    def reset_evaluation_data(self):
        """Delete the stored evaluation data."""
        self.data: Dict[str, np.ndarray] = {
            "users": np.zeros(0, int),
            "losses": np.zeros(0, float),
            "seconds": np.zeros(0, int),
            "red_flags": np.zeros(0, bool),
            "skipped": np.zeros(0, bool),
        }
        self.index = 0
        self.token_accuracy = torch.tensor(0, dtype=torch.float)
        self.token_count = torch.tensor(0, dtype=torch.long)
        self.eval_loss = torch.tensor(0, dtype=torch.float)
        self.eval_lines_count = torch.tensor(0, dtype=torch.long)
        if Application.instance().using_cuda:
            self.token_accuracy = self.token_accuracy.cuda()
            self.token_count = self.token_count.cuda()
            self.eval_loss = self.eval_loss.cuda()
            self.eval_lines_count = self.eval_lines_count.cuda()

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
            self.data[key] = self.data[key][: self.index]
        # Check if the data is sorted
        if not np.all(np.diff(self.data["seconds"]) >= 0):
            # Sort the data by seconds
            sorted_indices = np.argsort(self.data["seconds"])
            for key in self.data:
                self.data[key] = self.data[key][sorted_indices]
        # Compute final test loss
        self.eval_loss = self.eval_loss / max(self.eval_lines_count, 1)
        self.eval_lines_count += 1 - self.eval_lines_count  # Reset to 1, keep the same Tensor, type and device
        # Prepared the normalised losses
        self._normalise_losses()

        self.data_is_prepared = True

    def _normalise_losses(self):
        """Performs user-level anomaly score normalization by subtracting the
        average anomaly score of the user from each event (log line).

        Mainly relevant to word tokenization
        """
        # Loop over every user
        average_losses_user = np.ones_like(self.data["losses"])
        for user in tqdm(np.unique(self.data["users"]), desc="Normalising (user)"):
            user_indices = self.data["users"] == user
            # Compute the average loss for this user
            average_loss = np.average(self.data["losses"][user_indices])
            average_losses_user[user_indices] = average_loss
        # Apply the normalization
        self.data["normalised_losses"] = self.data["losses"] - average_losses_user

    def get_metrics(self):
        """Computes and returns all baseline metrics."""
        metrics_at_chosen_thresholds = self.get_metrics_at_chosen_thresholds()
        return_dict = {
            "eval/loss": self.get_test_loss(),
            "eval/token_accuracy": self.get_token_accuracy(),
            "eval/token_perplexity": self.get_token_perplexity(),
            "eval/AUC": self.get_auc_score(),
            "eval/AP": self.get_ap_score(),
            "eval/total_lines": len(self.data["losses"]),
            "eval/total_reds": np.sum(self.data["red_flags"]),
            "eval/skipped_lines": np.sum(self.data["skipped"]),
            "eval/skipped_reds": np.sum(self.data["red_flags"][self.data["skipped"]]),
        }
        return_dict.update(metrics_at_chosen_thresholds)
        return return_dict

    def get_test_loss(self):
        """Returns the accuracy of the model token prediction."""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        return self.eval_loss.item()

    def get_token_accuracy(self):
        """Returns the accuracy of the model token prediction."""
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        return self.token_accuracy.item()

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

    def get_metrics_at_chosen_thresholds(self, normalised=False):
        """Computes TPR/FPR/Precision/Recall at two thresholds:
        - FPR at above 0.1%
        - Highest Precision achieved (with Recall >= 0.1)
        """
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        # Get the relevant data - normalised or not
        losses = self.data["normalised_losses"] if normalised else self.data["losses"]
        fp_rate, tp_rate, roc_thresholds = metrics.roc_curve(self.data["red_flags"], losses, pos_label=1)
        precisions, recalls, pr_thresholds = metrics.precision_recall_curve(self.data["red_flags"], losses, pos_label=1)

        # FPR at 0.1%
        acceptable_fpr_index = np.where(fp_rate==np.min(fp_rate[fp_rate>0.001]))
        acceptable_tpr = tp_rate[acceptable_fpr_index]
        acceptable_fpr = fp_rate[acceptable_fpr_index]
        acceptable_threshold = roc_thresholds[acceptable_fpr_index]
        acceptable_recall = recalls[np.where(pr_thresholds==acceptable_threshold)]
        acceptable_precision = precisions[np.where(pr_thresholds==acceptable_threshold)]

        recall_low_bound_index = np.where(recalls==np.min(recalls[recalls>=0.1]))
        peak_precision_index = np.where(precisions==np.max(precisions[:recall_low_bound_index+1]))
        peak_precision = precisions[peak_precision_index]
        peak_precision_recall = recalls[peak_precision_index]
        peak_precision_threshold = pr_thresholds[peak_precision_index]
        peak_precision_tpr = tp_rate[np.where(roc_thresholds==peak_precision_threshold)]
        peak_precision_fpr = fp_rate[np.where(roc_thresholds==peak_precision_threshold)]
        return_dict = {
            "eval/0.1p_fpr": acceptable_fpr,
            "eval/0.1p_fpr_tpr": acceptable_tpr,
            "eval/0.1p_fpr_threshold": acceptable_threshold,
            "eval/0.1p_fpr_precision": acceptable_precision,
            "eval/0.1p_fpr_recall": acceptable_recall,
            "eval/peak_precision": peak_precision,
            "eval/peak_precision_recall": peak_precision_recall,
            "eval/peak_precision_threshold": peak_precision_threshold,
            "eval/peak_precision_fpr": peak_precision_fpr,
            "eval/peak_precision_tpr": peak_precision_tpr,
        }
        return return_dict

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
        percentiles=(75, 95, 99),
        smoothing=1,
        colors=("darkorange", "gold"),
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

    def plot_roc_curve(self, title="ROC", normalised=False, nonskipped=False):
        """Plots the ROC (Receiver Operating Characteristic) curve, i.e. TP-FP
        tradeoff.

        Also returns the corresponding auc score.
        """
        if not self.data_is_prepared:
            self.prepare_evaluation_data()
        auc_score = self.get_auc_score(normalised=normalised)

        # Get the relevant data - normalised or not
        losses = self.data["normalised_losses"] if normalised else self.data["losses"]
        red_flags = self.data["red_flags"]
        if nonskipped:
            losses = losses[self.data["skipped"] == 0]
            red_flags = red_flags[self.data["skipped"] == 0]

        full_fp_rate, full_tp_rate, _ = metrics.roc_curve(red_flags, losses, pos_label=1)
        # Scale fp_rate, tp_rate down to contain <2'000 values
        # E.g. if original length is 1'000'000, only take every 500th value
        step_size = (len(full_fp_rate) // 2000) + 1
        fp_rate = full_fp_rate[::step_size]
        tp_rate = full_tp_rate[::step_size]
        # Ensure the last value in full_fp_rate and full_tp_rate is included
        if fp_rate[-1] != full_fp_rate[-1]:
            fp_rate = np.append(fp_rate, full_fp_rate[-1])
            tp_rate = np.append(tp_rate, full_tp_rate[-1])
        # Y value (tp_rate) is monotonically increasing in ROC curves, so ignore any values after we reach 1.0 tp_rate
        last_index = np.where(np.isclose(tp_rate, 1.0))[0][0]
        fp_rate = fp_rate[: last_index + 1]
        tp_rate = tp_rate[: last_index + 1]
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

        # Plot using scikit-learn and matplotlib
        # red_flag_count = sum(self.data["red_flags"])
        # non_red_flag_count = len(self.data["red_flags"]) - red_flag_count
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

        # Scale precision, recall down to contain <2'000 values
        # E.g. if original length is 1'000'000, only take every 500th value
        step_size = (len(full_precision) // 2000) + 1
        precision = full_precision[::step_size]
        recall = full_recall[::step_size]

        # Ensure the last value in full_precision and full_recall is included
        if precision[-1] != full_precision[-1]:
            precision = np.append(precision, full_precision[-1])
            recall = np.append(recall, full_recall[-1])
        # Erase the full fp and tp lists
        full_precision = full_recall = []

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

        # Plot using scikit-learn and matplotlib
        xlabel = "Recall"
        ylabel = "Precision"
        plt.plot(
            recall,
            precision,
            color="orange",
            lw=2,
            label="Intrusion events",
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        return AP_score, plt

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

        # get nonskipped roc curve
        evaluator_metrics["eval/AUC_nonskipped"], nonskipped_roc_plot = self.plot_roc_curve(title="ROC (nonskipped)", nonskipped=True)
        if self.use_wandb:
            wandb.log({"ROC Curve (nonskipped)": nonskipped_roc_plot})

        # get pr curve
        evaluator_metrics["eval/AP"], pr_plot = self.plot_pr_curve()
        if self.use_wandb:
            wandb.log({"PR Curve": pr_plot})

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
            wandb.log({"PR Curve (normalised)": pr_plot})

        # Log the evaluation results
        if self.use_wandb and wandb.run is not None:
            for key, value in evaluator_metrics.items():
                wandb.run.summary[key] = value
        return evaluator_metrics
