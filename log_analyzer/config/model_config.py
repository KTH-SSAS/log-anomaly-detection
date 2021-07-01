from log_analyzer.config.config import Config


class LSTMConfig(Config):
    """Configuration class for LSTM models"""

    def __init__(self, layers, vocab_size, embedding_dim, attention_type, attention_dim, sequence_length=None) -> None:
        super().__init__()
        self.layers = layers
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

    @property
    def input_dim(self):
        """Feature length of input to LSTM"""
        return self.embedding_dim


class TieredLSTMConfig(LSTMConfig):
    """Configuration class for LSTM models"""

    def __init__(self, context_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_layers = context_layers

    @property
    def input_dim(self):
        """Feature length of input to LSTM"""
        return self.embedding_dim + self.context_layers[-1]
