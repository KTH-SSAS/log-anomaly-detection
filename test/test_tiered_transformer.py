import pytest
import torch

from log_analyzer.config.model_config import TieredTransformerConfig, TransformerConfig
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
CTX_LV_MODEL_DIM = 64
CTX_LV_FFW_DIM = 100
LEN_SAVED_HISTORY = 10
NUM_USERS = 100


@pytest.fixture
def test_config():
    context_config = TransformerConfig(LAYERS, FFW_DIM, MODEL_DIM, ATTENTION_HEAD, DROPOUT_RATE)
    config = TieredTransformerConfig(
        LAYERS, FFW_DIM, MODEL_DIM, ATTENTION_HEAD, DROPOUT_RATE, context_config, SHIFT_WINDOW
    )
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
    tieredTransformer = TieredTransformer(test_config, bidirectional=False)
    token_output, loss = tieredTransformer(test_input, context_history)

    return (
        torch.all(
            torch.ones(BATCH_SIZE) + CONSECUTIVE_LOG == tieredTransformer.saved_context_history_lengths[test_input[0]]
        )
        and tieredTransformer.saved_context_histories.shape == torch.Size([NUM_USERS, SHIFT_WINDOW + 1, VOCAB_SIZE])
        and token_output.shape == torch.Size([CONSECUTIVE_LOG, BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE])
    )
