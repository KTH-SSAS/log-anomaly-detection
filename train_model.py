from argparse import ArgumentParser

from log_analyzer.train_model import main

"""
Entrypoint script for training
Example:
train_model.py
--model-type
lstm,
--model-config,
config/lanl_char_config_model.json,
--trainer-config,
config/config_trainer.json,
--data-config,
config/lanl_char_config_data.json,
--data-folder,
data/data_examples/raw_day_split,
--bidir
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-type", choices=["lstm", "tiered-lstm", "transformer", "tiered-transformer"], required=True
    )
    parser.add_argument("--model-config", type=str, help="Model configuration file.", required=True)
    parser.add_argument("--data-config", type=str, help="Data description file.", required=True)
    parser.add_argument("--data-folder", type=str, help="Path to data files.", required=True)
    parser.add_argument("--trainer-config", type=str, help="Trainer configuration file.", required=True)
    parser.add_argument("--load-from-checkpoint", type=str, help="Checkpoint to resume training from")
    parser.add_argument(
        "--bidir", dest="bidirectional", action="store_true", help="Whether to use bidirectional lstm for lower tier."
    )
    parser.add_argument("--model-dir", type=str, help="Directory to save stats and checkpoints to", default="runs")
    parser.add_argument(
        "--eval_model",
        action="store_true",
        help="Including this option will run the model through standard evaluation and return appropriate metrics and plots.",
    )
    parser.add_argument(
        "--wandb_sync", action="store_true", help="Including this option will sync the wandb data with the cloud."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA acceleration for training.")
    args = parser.parse_args()
    main(args)
