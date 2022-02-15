import json
from dataclasses import dataclass
from typing import List, Optional

from log_analyzer.config.config import Config


@dataclass
class ModelConfig(Config):
    sequence_length: Optional[int]


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


@dataclass
class TieredTransformerConfig(TransformerConfig):
    """Configuration class for Tiered Transformer models."""

    context_config: TransformerConfig
    shift_window: int

    @property
    def input_dim(self):
        """Feature length of input to LSTM."""
        return self.model_dim + self.context_config.model_dim

    @classmethod
    def init_from_file(cls, filename):
        with open(filename, "r") as f:
            data: dict = json.load(f)

        data["context_config"] = TransformerConfig(
            **data["context_config"], vocab_size=data["vocab_size"], sequence_length=data["sequence_length"]
        )

        return cls.init_from_dict(data)

@dataclass
class LoglineTransformerConfig(TransformerConfig):
    """Configuration class for Logline Transformer models."""

    window_size: int
