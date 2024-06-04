"""Functions to test different model configurations."""

from pathlib import Path

import pytest
import torch

from log_analyzer.application import Application
from log_analyzer.train_loop import eval_model, init_from_config_classes, train_model

from . import utils


@pytest.mark.parametrize("model_type", ["lstm", "tiered-lstm", "transformer", "multiline-transformer"])
@pytest.mark.parametrize("cuda", [True, False])
def test_evaluator(tmpdir, model_type, cuda):
    if cuda and not torch.cuda.is_available():
        pytest.skip()

    Application.instance()._use_cuda = cuda  # pylint: disable=protected-access
    bidir = False
    token_level = "word-fields"

    args = utils.set_args(bidir, model_type, token_level)
    args["base_logdir"] = Path(tmpdir) / "runs"

    # Use a reasonable batch size (for testing)
    args["trainer_config"].train_batch_size = 10
    args["trainer_config"].eval_batch_size = 10
    if model_type == "multiline-transformer":
        # Use a reasonable window size (for testing)
        args["model_config"].shift_window = 3

    trainer, evaluator, train_loader, val_loader, test_loader = init_from_config_classes(**args)
    _ = train_model(trainer, train_loader, val_loader)
    _ = eval_model(evaluator, test_loader, store_eval_data=True)

    # Run through complete evaluator functionality and get numerical metrics
    metrics = evaluator.run_all()
    print(metrics)
    assert evaluator.data_is_prepared

    assert metrics["eval/token_accuracy"] >= 0 and metrics["eval/token_accuracy"] <= 1
    assert metrics["eval/token_perplexity"] >= 1
    assert metrics["eval/AUC"] >= 0 and metrics["eval/AUC"] <= 1
