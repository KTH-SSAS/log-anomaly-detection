from argparse import ArgumentParser
from log_analyzer.train_loop import init_from_args, train_model

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
    # Create the trainer+model
    trainer, train_loader, test_loader = init_from_args(args)
    # Train the model
    train_model(trainer, train_loader, test_loader)


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
    args = parser.parse_args()
    main(args)