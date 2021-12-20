import logging
import os
from argparse import ArgumentParser

import torch

import wandb
from log_analyzer.application import Application
from log_analyzer.train_loop import eval_model, init_from_args, train_model

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


def prepare_args():
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
        "--no-eval-model",
        action="store_true",
        help="Including this option will skip running the model through standard evaluation and returning appropriate metrics and plots.",
    )
    parser.add_argument(
        "--wandb-sync", action="store_true", help="Including this option will sync the wandb data with the cloud."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA acceleration for training.")
    args = parser.parse_args()
    return args


def main():

    args = prepare_args()

    #  Start a W&B run

    os.environ["WANDB_MODE"] = "online" if args.wandb_sync else "offline"

    wandb.init(project="logml", entity="log-data-ml", config=args)
    wandb_initalized = True

    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA not available. Ignoring the --cuda option.")
        cuda = False
    else:
        cuda = args.use_cuda

    Application(cuda=cuda, wandb=wandb_initalized)

    if args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    logging.basicConfig(level=log_level)

    # Create the trainer+model
    trainer, train_loader, val_loader, test_loader = init_from_args(args)
    # Train the model
    train_model(trainer, train_loader, val_loader)
    # Test the model
    eval_model(trainer, test_loader, store_eval_data=(not args.no_eval_model))

    # Perform standard evaluation on the model
    if Application.instance().wandb_initialized and not args.no_eval_model:
        trainer.evaluator.run_all()


if __name__ == "__main__":
    main()
