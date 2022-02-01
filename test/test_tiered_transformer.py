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

@pytest.fixture
def context_input():
    return torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, CTX_LV_FFW_DIM))

@pytest.fixture
def context_history():
    return torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, LEN_SAVED_HISTORY, VOCAB_SIZE)) 

def test_tiered_transformer_forward_word(test_config : TieredTransformerConfig, 
                                        test_input, context_input, context_history_input):
    tieredTransformer = TieredTransformer(test_config)
    tag_output, ctxt_vector, ctx_history_output = tieredTransformer(test_input, context_input, context_history_input)
    return (ctx_history_output[:,:-3,:] == context_history_input[:,3:,:]).all() and \
            ctx_history_output.shape == torch.Size([BATCH_SIZE, SHIFT_WINDOW, VOCAB_SIZE]) and \
            tag_output.shape == torch.Size([CONSECUTIVE_LOG, BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE])