"""Functions to test different model configurations."""
from pathlib import Path

import pytest
import torch.cuda

from . import utils


@pytest.mark.parametrize("tokenization", ["word-field", "word-global", "char"])
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

    utils.run_test(args)
    assert True


@pytest.mark.parametrize("tokenization", ["word-field", "word-global", "char"])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("model_type", ["transformer", "tiered-transformer"])
def test_transformer(tmpdir, model_type, bidirectional, tokenization):

    args = utils.set_args(bidirectional, model_type, tokenization)
    if model_type == "tiered-transformer":
        # Reduce batch size to not immediately flush.
        args["trainer_config"].train_batch_size = 10
        args["trainer_config"].eval_batch_size = 10
    args["base_logdir"] = Path(tmpdir) / "runs"

    utils.run_test(args)
    assert True
