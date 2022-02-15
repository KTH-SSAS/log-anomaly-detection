import pytest
import torch

from log_analyzer.config.model_config import TransformerConfig
from log_analyzer.model.transformer import Transformer

SEQUENCE_LENGTH = 10
VOCAB_SIZE = 128
BATCH_SIZE = 64


@pytest.fixture
def test_config():
    return TransformerConfig(None, SEQUENCE_LENGTH, 64, 64, 2, VOCAB_SIZE, 0.1)


@pytest.fixture
def test_input():
    return torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQUENCE_LENGTH))


def test_forward(test_config, test_input):
    transformer = Transformer(test_config)
    # Provide test_input as targets (incorrect, but lets us test compute_loss)
    output, _ = transformer(test_input, targets=test_input)
    assert output.shape == torch.Size([BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE])
    pass
