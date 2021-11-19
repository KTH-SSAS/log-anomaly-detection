"""Functions to test different model configurations."""
import os

import pytest
import torch.cuda
import utils


@pytest.mark.parametrize("tokenization", ["word", "char"])
@pytest.mark.parametrize("bidir", [True, False])
@pytest.mark.parametrize("model_type", ["lstm", "tiered-lstm"])
@pytest.mark.parametrize("cuda", [True, False])
def test_lstm(tmpdir, model_type, bidir, tokenization, cuda):

    if cuda and not torch.cuda.is_available():
        pytest.skip()

    args = utils.set_args(bidir, model_type, tokenization)
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True


# TODO make the char test not fail
@pytest.mark.parametrize("tokenization", ["word", "char"])
@pytest.mark.parametrize("model_type", ["transformer", "tiered-transformer"])
def test_transformer(tmpdir, model_type, tokenization):

    args = utils.set_args(False, model_type, tokenization)
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True
