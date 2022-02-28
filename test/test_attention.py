import os

import pytest
import utils


@pytest.mark.parametrize("tokenization", ["word", "char"])
@pytest.mark.parametrize("bidir", [True, False])
@pytest.mark.parametrize("attention_type", ["fixed", "semantic"])
def test_attention(tmpdir, tokenization, bidir, attention_type):

    args = utils.set_args(bidir, "lstm", tokenization)
    args["model_config"].attention_type = attention_type
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True


@pytest.mark.parametrize(
    "tokenization,bidir",
    [
        ("word", True),
        ("word", False),
        pytest.param("char", False, marks=pytest.mark.xfail),
        pytest.param("char", True, marks=pytest.mark.xfail),
    ],
)
def test_syntax_attention(tmpdir, tokenization, bidir):
    bidir = False
    model_type = "lstm"

    args = utils.set_args(bidir, model_type, tokenization)
    args["model_config"].attention_type = "syntax"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True
