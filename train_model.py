from argparse import ArgumentParser
from log_analyzer.train_loop import init_from_args, train_model
from eval_model import eval_model
import log_analyzer.application as application
import wandb
import os
import logging
import torch

"""
Entrypoint script for training
Example:
train_model.py
--model-type
lstm,
--model-config,
config/lanl_char_config_model.json,
--trainer-config,
config/lanl_char_config_trainer.json,
--data-folder,
data/data_examples/raw_day_split,
--bidir
"""

def main(args):

    #  Start a W&B run

    os.environ['WANDB_MODE'] = 'online' if args.wandb_sync else 'offline'

    wandb.init(project='logml', entity='log-data-ml', config=args)
    wandb_initalized = True

    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA not available. Ignoring the --cuda option.")
        cuda = False
    else:
        cuda = args.use_cuda

    application.Application(cuda=cuda, wandb=wandb_initalized)

    if args.verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    logging.basicConfig(level=log_level)

    # Create the trainer+model
    trainer, train_loader, test_loader = init_from_args(args)
    # Train the model
    train_model(trainer, train_loader, test_loader, store_eval_data=args.eval_model)

    # Perform standard evaluation on the model
    if args.eval_model and application.wandb_initalized:
        eval_model(trainer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model-type', choices=['lstm', 'tiered-lstm', 'transformer'], required=True)
    parser.add_argument('--model-config', type=str, help="Model configuration file.", required=True)
    parser.add_argument('--data-config', type=str, help="Data description file.", required=True)
    parser.add_argument("--data-folder", type=str, help="Path to data files.", required=True)
    parser.add_argument('--trainer-config', type=str, help="Trainer configuration file.", required=True)
    parser.add_argument('--load-from-checkpoint', type=str, help='Checkpoint to resume training from')
    parser.add_argument('--bidir', dest='bidirectional', action='store_true',
                        help='Whether to use bidirectional lstm for lower tier.')
    parser.add_argument('--model-dir', type=str, help='Directory to save stats and checkpoints to', default='runs')
    parser.add_argument('--eval_model', action='store_true', help="Including this option will run the model through standard evaluation and return appropriate metrics and plots.")
    parser.add_argument('--wandb_sync', action='store_true', help="Including this option will sync the wandb data with the cloud.")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true', help="Use CUDA acceleration for training.")
    args = parser.parse_args()
    main(args)