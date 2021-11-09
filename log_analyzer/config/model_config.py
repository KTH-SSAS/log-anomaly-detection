from dataclasses import dataclass
from typing import List

from log_analyzer.config.config import Config


@dataclass
class ModelConfig(Config):
    sequence_length: int


@dataclass
class LSTMConfig(ModelConfig):
    """Configuration class for LSTM models."""

    layers: List[int]
    attention_type: str
    attention_dim: int
    embedding_dim: int
    vocab_size: int

    @property
    def input_dim(self):
        """Feature length of input to LSTM."""
        return self.embedding_dim


@dataclass
class TieredLSTMConfig(LSTMConfig):
    """Configuration class for LSTM models."""

    context_layers: list

    @property
    def input_dim(self):
        """Feature length of input to LSTM."""
        return self.embedding_dim + self.context_layers[-1]

class TransformerConfig(Config):
    """Configuration class for Transformer models"""

    def __init__(self, layers, feedforward_dim, model_dim, attention_heads, vocab_size, dropout, sequence_length=None):
        super().__init__()
        self.layers = layers
        self.feedforward_dim = feedforward_dim
        self.model_dim = model_dim
        self.attention_heads = attention_heads
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.sequence_length = sequence_length

class TieredTransformerConfig(TransformerConfig):
    """Configuration class for Tiered Transformer models"""

    def __init__(self, context_model_dim, context_layers, context_feedforward_dim, context_attention_heads, context_dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_model_dim = context_model_dim
        self.context_layers =  context_layers
        self.context_feedforward_dim = context_feedforward_dim
        self.context_attention_heads = context_attention_heads
        self.context_dropout = context_dropout

    @property
    def input_dim(self):
        """Feature length of input to LSTM"""
        return self.model_dim + self.context_model_dim
