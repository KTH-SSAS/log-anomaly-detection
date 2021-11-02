"""Functions to test different model configurations"""
import os

import pytest

import utils


def test_forward_word(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'word'
    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    train_losses, test_losses = utils.run_test(args)
    assert True


def test_forward_char(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    utils.run_test(args)
    assert True


def test_bidirectional_word(tmpdir):
    bidir = True
    model_type = 'lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    utils.run_test(args)
    assert True


def test_bidirectional_char(tmpdir):
    bidir = True
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    utils.run_test(args)
    assert True


def test_tiered_char(tmpdir):
    bidir = False
    model_type = 'tiered-lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    utils.run_test(args)
    assert True


def test_tiered_word(tmpdir):
    bidir = False
    model_type = 'tiered-lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    utils.run_test(args)
    assert True


def test_tiered_bidirectional_char(tmpdir):
    bidir = True
    model_type = 'tiered-lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    utils.run_test(args)
    assert True


def test_tiered_bidirectional_word(tmpdir):
    bidir = True
    model_type = 'tiered-lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    utils.run_test(args)
    assert True


def test_transformer_word(tmpdir):
    bidir = False
    model_type = 'transformer'
    token_level = 'word'
    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')
    utils.run_test(args)
    assert True
