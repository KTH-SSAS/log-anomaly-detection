from log_analyzer.config.config import Config

class LSTMConfig(Config):
    """Configuration class for LSTM models"""
    def __init__(self, layers, vocab_size, embedding_dim, bidirectional, attention_type, attention_dim, jagged) -> None:
        super().__init__()
        self.layers = layers
        self.bidirectional = bidirectional
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim
        self.vocab_size = vocab_size
        self.jagged = jagged

class TieredLSTMConfig(LSTMConfig):
    """Configuration class for LSTM models"""
    def __init__(self, layers, vocab_size, embedding_dim, bidirectional, attention_type, attention_dim, jagged, context_layers) -> None:
        super(TieredLSTMConfig, self).__init__(layers, vocab_size, embedding_dim, bidirectional, attention_type, attention_dim, jagged)
        self.context_layers = context_layers
        self.input_dim += context_layers[-1]
