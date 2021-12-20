"""Functions to test different model configurations."""
import os

import pytest
import utils

from log_analyzer.train_loop import eval_model, init_from_config_classes, train_model


@pytest.mark.parametrize("model_type", ["lstm", "tiered-lstm", "transformer"])
def test_evaluator(tmpdir, model_type):
    bidir = False
    token_level = "word"

    args = utils.set_args(bidir, model_type, token_level)
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    trainer, train_loader, val_loader, test_loader = init_from_config_classes(**args)
    _ = train_model(trainer, train_loader, val_loader)
    _ = eval_model(trainer, test_loader, store_eval_data=True)

    # Numerical metrics
    metrics = trainer.evaluator.get_metrics()
    assert trainer.evaluator.data_is_prepared

    assert metrics["eval/token_accuracy"] >= 0 and metrics["eval/token_accuracy"] <= 1
    assert metrics["eval/token_perplexity"] >= 1
    assert metrics["eval/AUC"] >= 0 and metrics["eval/AUC"] <= 1

    # Run through complete evaluator functionality
    trainer.evaluator.run_all()
