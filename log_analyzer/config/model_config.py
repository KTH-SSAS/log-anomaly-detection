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


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration class for Transformer models."""
    layers: int
    feedforward_dim: int
    model_dim: int
    attention_heads: int
    vocab_size: int
    dropout: float
