import pytest
import torch

from log_analyzer.config.model_config import TieredTransformerConfig, TransformerConfig
from log_analyzer.model.transformer import TieredTransformer

CONSECUTIVE_LOG = 3
SEQUENCE_LENGTH = 10
VOCAB_SIZE = 128
BATCH_SIZE = 64
SHIFT_WINDOW = 10
LOW_LV_MODEL_DIM = 64
LOW_LV_FFW_DIM = 64
DROPOUT_RATE = 0.1
ATTENTION_HEAD = 2
CTX_LV_MODEL_DIM = 64
CTX_LV_FFW_DIM = 100
LEN_SAVED_HISTORY = 10 

@pytest.fixture
def test_config():
    context_config = TransformerConfig(None, SEQUENCE_LENGTH, CTX_LV_MODEL_DIM, 
                                        CTX_LV_FFW_DIM, ATTENTION_HEAD, VOCAB_SIZE, DROPOUT_RATE)
    return TieredTransformerConfig(None, SEQUENCE_LENGTH, LOW_LV_FFW_DIM, 
                                    LOW_LV_MODEL_DIM, ATTENTION_HEAD, VOCAB_SIZE, DROPOUT_RATE, 
                                    context_config, SHIFT_WINDOW)

@pytest.fixture
def test_input():
    return torch.randint(low=0, high=VOCAB_SIZE, size=(CONSECUTIVE_LOG, BATCH_SIZE, SEQUENCE_LENGTH))

