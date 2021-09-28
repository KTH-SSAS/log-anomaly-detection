from log_analyzer.config.config import Config


class DataConfig(Config):

    def __init__(self, sentence_length, vocab_size, number_of_days, tokenization) -> None:
        super().__init__()
        self.tokenization = tokenization
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.number_of_days = number_of_days


class TrainerConfig(Config):

    def __init__(self, train_files, test_files, shuffle_train_data, batch_size, learning_rate, early_stopping,
                 early_stop_patience, scheduler_gamma, scheduler_step_size, mixed_precision) -> None:
        super().__init__()

        # Data settings
        self.train_files = train_files
        self.test_files = test_files
        self.shuffle_train_data = shuffle_train_data

        # Optimization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.early_stop_patience = early_stop_patience
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.mixed_precision = mixed_precision
