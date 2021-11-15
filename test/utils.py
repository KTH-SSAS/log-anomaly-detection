from log_analyzer.config.model_config import LSTMConfig, TieredLSTMConfig
from log_analyzer.config.trainer_config import DataConfig, TrainerConfig
from log_analyzer.train_loop import get_model_config, init_from_config_classes, train_model


def set_args(bidir, model_type, token_level):
    """Prepares a dictionary of settings that can be used for testing."""
    # Common args (defaults, can be changed)
    args = {}

    args["bidirectional"] = bidir
    args["model_type"] = model_type
    trainer_config = TrainerConfig.init_from_file("config/config_trainer.json")
    trainer_config.train_files = ["0.txt", "1.txt"]
    trainer_config.test_files = ["2.txt"]
    args["trainer_config"] = trainer_config

    if model_type == "tiered-lstm":
        model_config_file = f"config/lanl_{token_level}_config_model_tiered.json"

    elif model_type == "transformer":
        # TODO We should standardize the config file names
        model_config_file = f"config/lanl_{token_level}_config_{model_type}.json"

    else:
        model_config_file = f"config/lanl_{token_level}_config_model.json"

    args["model_config"] = get_model_config(model_config_file, model_type)

    args["data_folder"] = f"data/test_data/{token_level}_day_split"
    args["data_config"] = DataConfig.init_from_file(f"config/lanl_{token_level}_config_data.json")

    # Return the prepared args
    return args


def run_test(args):
    trainer, train_loader, val_loader, test_loader = init_from_config_classes(**args)
    train_losses, test_losses = train_model(trainer, train_loader, val_loader, test_loader, store_eval_data=False)
    return train_losses, test_losses
