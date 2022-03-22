import json
from dataclasses import dataclass, field
from typing import List, Optional

from log_analyzer.config.config import Config


@dataclass
class ModelConfig(Config):
    sequence_length: Optional[int] = field(init=False, default=None)
    _vocab_size: Optional[int] = field(init=False, default=None)

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("Vocab size was not set!")
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value


@dataclass
class LSTMConfig(ModelConfig):
    """Configuration class for LSTM models."""

    layers: List[int]
    attention_type: str
    attention_dim: int
    embedding_dim: int

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
    dropout: float


@dataclass
class TieredTransformerConfig(TransformerConfig):
    """Configuration class for Tiered Transformer models."""

    context_config: TransformerConfig
    shift_window: int
    _number_of_users: Optional[int] = field(init=False, default=None)

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("Vocab size was not set!")
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value
        self.context_config.vocab_size = value

    @property
    def number_of_users(self) -> int:
        if self._number_of_users is None:
            raise RuntimeError("Number of users was not set!")
        return self._number_of_users

    @number_of_users.setter
    def number_of_users(self, value):
        self._number_of_users = value

    @property
    def input_dim(self):
        """Feature length of input to LSTM."""
        return self.model_dim + self.context_config.model_dim

    @classmethod
    def init_from_file(cls, filename):
        with open(filename, "r", encoding="utf8") as f:
            data: dict = json.load(f)

        data["context_config"] = TransformerConfig(
            **data["context_config"],
        )

        return cls.init_from_dict(data)


@dataclass
class MultilineTransformerConfig(TransformerConfig):
    """Configuration class for Multiline Transformer models."""

    window_size: int
    memory_type: str
    sentence_embedding: str
    embeddings_path: str
