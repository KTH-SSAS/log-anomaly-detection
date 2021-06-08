"""Functions to test different model configurations"""
import os
import pytest
import utils
from log_analyzer.train_loop import create_model, train_model

def test_forward_word(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    trainer, train_loader, test_loader = create_model(**args)
    train_losses, test_losses = train_model(trainer, train_loader, test_loader)
    assert True

def test_forward_char(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)

    args['base_logdir'] = os.path.join(tmpdir, 'runs')
    trainer, train_loader, test_loader = create_model(**args)
    train_losses, test_losses = train_model(trainer, train_loader, test_loader)
    assert True

def test_bidirectional_word(tmpdir):
    bidir = True
    model_type = 'lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)

    args['base_logdir'] = os.path.join(tmpdir, 'runs')
    trainer, train_loader, test_loader = create_model(**args)
    train_losses, test_losses = train_model(trainer, train_loader, test_loader)
    assert True

def test_bidirectional_char(tmpdir):
    bidir = True
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)

    args['base_logdir'] = os.path.join(tmpdir, 'runs')
    trainer, train_loader, test_loader = create_model(**args)
    train_losses, test_losses = train_model(trainer, train_loader, test_loader)
    assert True

def test_tiered_char(tmpdir):
    bidir = False
    model_type = 'tiered-lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)

    args['base_logdir'] = os.path.join(tmpdir, 'runs')
    trainer, train_loader, test_loader = create_model(**args)
    train_losses, test_losses = train_model(trainer, train_loader, test_loader)
    assert True
    
def test_tiered_word(tmpdir):
    bidir = False
    model_type = 'tiered-lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)

    args['base_logdir'] = os.path.join(tmpdir, 'runs')
    trainer, train_loader, test_loader = create_model(**args)
    train_losses, test_losses = train_model(trainer, train_loader, test_loader)
    assert True