"""Functions to test different model configurations"""
import os
import pytest
import utils
from log_analyzer.train_loop import init_from_config_classes, train_model

def run_test(args):
    trainer, train_loader, test_loader = init_from_config_classes(**args)
    train_losses, test_losses = train_model(trainer, train_loader, test_loader)
    return train_losses, test_losses

def test_forward_word(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'word'
    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    train_losses, test_losses = run_test(args)
    assert True

def test_forward_char(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True

def test_forward_char_attention(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['model_config'].attention_type = 'fixed'
    args['model_config'].attention_dim = 10
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True

def test_forward_word_attention(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['model_config'].attention_type = 'fixed'
    args['model_config'].attention_dim = 10
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True
    
def test_forward_char_syntax_attention(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['model_config'].attention_type = 'syntax'
    args['model_config'].attention_dim = 10
    args['base_logdir'] = os.path.join(tmpdir, 'runs')
    with pytest.raises(RuntimeError) as execinfo:
        run_test(args)
    assert "sequence length has to bet set" in str(execinfo.value)
    assert True

def test_forward_word_syntax_attention(tmpdir):
    bidir = False
    model_type = 'lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['model_config'].attention_type = 'syntax'
    args['model_config'].attention_dim = 10
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True

def test_bidirectional_word(tmpdir):
    bidir = True
    model_type = 'lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True

def test_bidirectional_char(tmpdir):
    bidir = True
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True

def test_bidirectional_word_attention(tmpdir):
    bidir = True
    model_type = 'lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['model_config'].attention_type = 'fixed'
    args['model_config'].attention_dim = 10
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True

def test_bidirectional_char_attention(tmpdir):
    bidir = True
    model_type = 'lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['model_config'].attention_type = 'fixed'
    args['model_config'].attention_dim = 10
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True

def test_tiered_char(tmpdir):
    bidir = False
    model_type = 'tiered-lstm'
    token_level = 'char'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True
    
def test_tiered_word(tmpdir):
    bidir = False
    model_type = 'tiered-lstm'
    token_level = 'word'

    args = utils.set_args(bidir, model_type, token_level)
    args['base_logdir'] = os.path.join(tmpdir, 'runs')

    run_test(args)
    assert True