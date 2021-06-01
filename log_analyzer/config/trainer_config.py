from log_analyzer.config.config import Config


class DataConfig(Config):

    def __init__(self, train_files, test_files, sentence_length, vocab_size, number_of_days) -> None:
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.number_of_days = number_of_days

class TrainerConfig(Config):

    def __init__(self, data_config, batch_size, skipsos, jagged, bidirectional, learning_rate, early_stopping, 
    early_stop_patience, scheduler_gamma, scheduler_step_size) -> None:
        super().__init__()

        # Data settings
        self._data_config = data_config.__dict__ if data_config is DataConfig else data_config # Store this as a dict to facilitate serialization

        # Model settings
        self.bidirectional = bidirectional

        # Optimization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.early_stop_patience = early_stop_patience
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.jagged = jagged
        self.skipsos = skipsos

    @property
    def data_config(self) -> DataConfig:
        return DataConfig.init_from_dict(self._data_config)
