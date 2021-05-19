
from argparse import Namespace
from log_analyzer.config.config import Config


class TrainerConfig(Config):

    def __init__(self, batch_size, jagged, bidirectional, tiered, learning_rate, early_stopping, 
    early_stop_patience, scheduler_gamma, scheduler_step_size) -> None:
        super().__init__()

        # Model settings
        self.tiered = tiered
        self.bidirectional = bidirectional

        # Optimization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.es_patience = early_stop_patience
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.jagged = jagged
