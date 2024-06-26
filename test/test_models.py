"""Functions to test different model configurations."""

from pathlib import Path

import numpy as np
import pytest
import torch.cuda

from . import utils


@pytest.mark.parametrize("tokenization", ["word-fields", "word-global", "char"])
@pytest.mark.parametrize("bidir", [True, False])
@pytest.mark.parametrize("model_type", ["lstm", "tiered-lstm"])
@pytest.mark.parametrize("cuda", [True, False])
def test_lstm(tmpdir, model_type, bidir, tokenization, cuda):

    if cuda and not torch.cuda.is_available():
        pytest.skip()

    args = utils.set_args(bidir, model_type, tokenization)
    args["base_logdir"] = Path(tmpdir) / "runs"

    if model_type == "tiered-lstm":
        # Reduce batch size to not immediately flush.
        args["trainer_config"].train_batch_size = 10
        args["trainer_config"].eval_batch_size = 10

    utils.run_test(args, cuda)
    assert True


@pytest.mark.parametrize("tokenization", ["word-fields", "word-global", "char"])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("model_type", ["transformer", "tiered-transformer"])
@pytest.mark.parametrize("cuda", [True, False])
def test_transformer(tmpdir, model_type, bidirectional, tokenization, cuda):

    if cuda and not torch.cuda.is_available():
        pytest.skip()

    if bidirectional and (model_type == "tiered-transformer" or tokenization == "char"):
        # Bidirectional transformer is not supported for tiered-transformer or char tokenization.
        pytest.skip()

    args = utils.set_args(bidirectional, model_type, tokenization)
    if model_type == "tiered-transformer":
        # Reduce batch size to not immediately flush.
        args["trainer_config"].train_batch_size = 10
        args["trainer_config"].eval_batch_size = 10
    args["base_logdir"] = Path(tmpdir) / "runs"

    utils.run_test(args, cuda)
    assert True


@pytest.mark.parametrize("tokenization", ["word-fields", "word-global", "word-merged"])
@pytest.mark.parametrize("cuda", [True, False])
def test_multiline_transformer(tmpdir, tokenization, cuda):

    if cuda and not torch.cuda.is_available():
        pytest.skip()

    args = utils.set_args(False, "multiline-transformer", tokenization)
    args["trainer_config"].train_batch_size = 10
    args["trainer_config"].eval_batch_size = 10
    # Use a reasonable window size (for testing)
    args["model_config"].shift_window = 3
    args["base_logdir"] = Path(tmpdir) / "runs"

    train_losses, test_losses = utils.run_test(args, cuda)
    train_nans = np.any(np.isnan(train_losses))
    test_nans = np.any(np.isnan(test_losses))
    assert not (train_nans and test_nans), "NaNs in both train and test losses"
    assert not train_nans, "NaNs in train losses"
    assert not test_nans, "NaNs in test losses"
