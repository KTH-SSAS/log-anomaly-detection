import os

import pytest
import utils

from log_analyzer.train_loop import init_from_config_classes, train_model


def test_forward_char_attention(tmpdir):
    bidir = False
    model_type = "lstm"
    token_level = "char"

    args = utils.set_args(bidir, model_type, token_level)
    args["model_config"].attention_type = "fixed"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True


def test_forward_word_attention(tmpdir):
    bidir = False
    model_type = "lstm"
    token_level = "word"

    args = utils.set_args(bidir, model_type, token_level)
    args["model_config"].attention_type = "fixed"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True


def test_forward_char_syntax_attention(tmpdir):
    bidir = False
    model_type = "lstm"
    token_level = "char"

    args = utils.set_args(bidir, model_type, token_level)
    args["model_config"].attention_type = "syntax"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")
    with pytest.raises(RuntimeError) as execinfo:
        utils.run_test(args)
    assert "sequence length has to bet set" in str(execinfo.value)
    assert True


def test_forward_word_syntax_attention(tmpdir):
    bidir = False
    model_type = "lstm"
    token_level = "word"

    args = utils.set_args(bidir, model_type, token_level)
    args["model_config"].attention_type = "syntax"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True


def test_forward_char_semantic_attention(tmpdir):
    bidir = False
    model_type = "lstm"
    token_level = "char"

    args = utils.set_args(bidir, model_type, token_level)
    args["model_config"].attention_type = "semantic"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")
    utils.run_test(args)
    assert True


def test_forward_word_semantic_attention(tmpdir):
    bidir = False
    model_type = "lstm"
    token_level = "word"

    args = utils.set_args(bidir, model_type, token_level)
    args["model_config"].attention_type = "semantic"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True


def test_bidirectional_word_attention(tmpdir):
    bidir = True
    model_type = "lstm"
    token_level = "word"

    args = utils.set_args(bidir, model_type, token_level)
    args["model_config"].attention_type = "fixed"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True


def test_bidirectional_char_attention(tmpdir):
    bidir = True
    model_type = "lstm"
    token_level = "char"

    args = utils.set_args(bidir, model_type, token_level)
    args["model_config"].attention_type = "fixed"
    args["model_config"].attention_dim = 10
    args["base_logdir"] = os.path.join(tmpdir, "runs")

    utils.run_test(args)
    assert True
