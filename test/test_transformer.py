import torch

from log_analyzer.model.transformer import Transformer

SEQUENCE_LENGTH = 10
VOCAB_SIZE = 128
BATCH_SIZE = 64


def test_forward(test_config, test_input):
    transformer = Transformer(test_config, bidirectional=False)
    # Provide test_input as targets (incorrect, but lets us test compute_loss)
    output, _ = transformer(test_input, targets=test_input)
    assert output.shape == torch.Size([BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE])
