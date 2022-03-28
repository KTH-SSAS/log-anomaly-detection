import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from log_analyzer.config.model_config import TieredTransformerConfig
from log_analyzer.model.transformer import TieredTransformer

CONSECUTIVE_LOG = 3
SEQUENCE_LENGTH = 10
VOCAB_SIZE = 128
BATCH_SIZE = 64
SHIFT_WINDOW = 10
LAYERS = 2
MODEL_DIM = 64
FFW_DIM = 64
DROPOUT_RATE = 0.1
ATTENTION_HEAD = 2
LEN_SAVED_HISTORY = 10
NUM_USERS = 100


@pytest.fixture
def test_config():
    args = {
        "layers": LAYERS,
        "feedforward_dim": FFW_DIM,
        "model_dim": MODEL_DIM,
        "attention_heads": ATTENTION_HEAD,
        "dropout": DROPOUT_RATE,
        "shift_window": SHIFT_WINDOW,
    }
    config = TieredTransformerConfig(**args)
    config.vocab_size = VOCAB_SIZE
    config.number_of_users = NUM_USERS
    config.sequence_length = SEQUENCE_LENGTH
    return config


@pytest.fixture
def test_input():
    return (
        torch.randint(low=0, high=NUM_USERS, size=(BATCH_SIZE, 1)),
        torch.randint(low=0, high=VOCAB_SIZE, size=(CONSECUTIVE_LOG, BATCH_SIZE, SEQUENCE_LENGTH)),
    )


@pytest.fixture
def context_history():
    return torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, LEN_SAVED_HISTORY, VOCAB_SIZE))


def test_tiered_transformer_forward_word(test_config: TieredTransformerConfig, test_input, context_history):
    tieredTransformer = TieredTransformer(test_config(), bidirectional=False)
    ctx_lengths_before_run = pad_sequence(
        tieredTransformer.get_ctx_data(torch.squeeze(test_data[0]))[0], batch_first=True
    ).shape[0]
    token_output, _ = tieredTransformer(test_data, context_history)
    ctx_lengths_after_run = pad_sequence(
        tieredTransformer.get_ctx_data(torch.squeeze(test_data[0]))[0], batch_first=True
    ).shape[0]
    assert min(ctx_lengths_before_run + CONSECUTIVE_LOG, SHIFT_WINDOW) == ctx_lengths_after_run
    assert token_output.shape == torch.Size([CONSECUTIVE_LOG, BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE])
