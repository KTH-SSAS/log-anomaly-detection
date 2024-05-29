from log_analyzer.application import Application
from log_analyzer.config import TrainerConfig
from log_analyzer.train_loop import eval_model, get_model_config, init_from_config_classes, train_model


def set_args(bidir, model_type, token_level):
    """Prepares a dictionary of settings that can be used for testing."""
    # Common args (defaults, can be changed)
    args = {}

    args["bidirectional"] = bidir
    args["model_type"] = model_type
    args["tokenization"] = token_level
    args["counts_file"] = "data/counts.json"
    trainer_config = TrainerConfig.init_from_file("config/lanl_config_trainer.json")
    trainer_config.train_files = ["6.csv", "7.csv"]
    trainer_config.validation_files = ["6_two.csv"]
    trainer_config.test_files = ["8.csv"]
    args["trainer_config"] = trainer_config

    model_config_file = f"config/lanl_config_{model_type}.json"

    args["model_config"] = get_model_config(model_config_file, model_type)

    args["data_folder"] = "data/test_data"

    # Return the prepared args
    return args


def run_test(args, cuda=False):
    # this is not a great way to do this, but it's quick.
    Application.instance()._use_cuda = cuda  # pylint: disable=protected-access
    trainer, evaluator, train_loader, val_loader, test_loader = init_from_config_classes(**args)
    train_losses = train_model(trainer, train_loader, val_loader)
    test_losses = eval_model(evaluator, test_loader, store_eval_data=False)
    return train_losses, test_losses
