"""Functions to test different model configurations."""
import os

import pytest

from log_analyzer.train_loop import eval_model, init_from_config_classes, train_model

from . import utils


@pytest.mark.parametrize("model_type", ["lstm", "tiered-lstm", "transformer"])
def test_evaluator(tmpdir, model_type):
    bidir = False
    token_level = "word-field"

    args = utils.set_args(bidir, model_type, token_level)
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    if model_type == "tiered-lstm":
        # Reduce batch size to not immediately flush.
        args["trainer_config"].train_batch_size = 10
        args["trainer_config"].eval_batch_size = 10

    trainer, evaluator, train_loader, val_loader, test_loader = init_from_config_classes(**args)
    _ = train_model(trainer, train_loader, val_loader)
    _ = eval_model(evaluator, test_loader, store_eval_data=True)

    # Numerical metrics
    metrics = evaluator.get_metrics()
    assert evaluator.data_is_prepared

    assert metrics["eval/token_accuracy"] >= 0 and metrics["eval/token_accuracy"] <= 1
    assert metrics["eval/token_perplexity"] >= 1
    # assert metrics["eval/AUC"] >= 0 and metrics["eval/AUC"] <= 1

    # Run through complete evaluator functionality
    evaluator.run_all()
