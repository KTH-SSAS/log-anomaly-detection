"""Functions to test different model configurations."""
import os

import pytest
import utils

from log_analyzer.eval_model import eval_model
from log_analyzer.train_loop import init_from_config_classes, train_model


def test_evaluator_lstm(tmpdir):
    bidir = False
    model_type = "lstm"
    token_level = "word"

    args = utils.set_args(bidir, model_type, token_level)
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    trainer, train_loader, val_loader, test_loader = init_from_config_classes(**args)
    train_losses, test_losses = train_model(trainer, train_loader, val_loader, test_loader, store_eval_data=True)

    # Numerical metrics
    metrics = trainer.evaluator.get_metrics()
    assert trainer.evaluator.data_is_prepared

    assert metrics["eval/token_accuracy"] >= 0 and metrics["eval/token_accuracy"] <= 1
    assert metrics["eval/token_perplexity"] >= 1
    assert metrics["eval/AUC"] >= 0 and metrics["eval/AUC"] <= 1

    # Line loss percentiles plot
    for percentiles in [[75, 95, 99], [75, 95], [50, 75, 95, 99]]:
        for smoothing in [0, 1, 10, 31]:
            for outliers in [0, 0.5, 1, 60]:
                trainer.evaluator.plot_line_loss_percentiles(
                    percentiles=percentiles,
                    smoothing=smoothing,
                    colors=["purple", "darkblue", "blue", "skyblue"],
                    ylim=(-1, -1),
                    outliers=outliers,
                    legend=smoothing % 2 == 0,
                )

    # ROC curve plot
    auc_score, _ = trainer.evaluator.plot_roc_curve()
    assert auc_score > 0 and auc_score < 1

    # PR curve plot
    ap_score, _ = trainer.evaluator.plot_pr_curve()
    assert ap_score > 0 and ap_score < 1

    # Data normalisation
    assert not trainer.evaluator.data_is_normalized
    trainer.evaluator.normalize_losses()
    assert trainer.evaluator.data_is_normalized


def test_evaluator_tiered(tmpdir):
    bidir = True
    model_type = "tiered-lstm"
    token_level = "char"

    args = utils.set_args(bidir, model_type, token_level)
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    trainer, train_loader, val_loader, test_loader = init_from_config_classes(**args)
    # Only unique thing in the tiered version of the evaluator is the data-storing code
    train_losses, test_losses = train_model(trainer, train_loader, val_loader, test_loader, store_eval_data=True)
    eval_model(trainer)
    assert True
