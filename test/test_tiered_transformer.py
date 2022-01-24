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
