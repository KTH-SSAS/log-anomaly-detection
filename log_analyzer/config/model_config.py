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