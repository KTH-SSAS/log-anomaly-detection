from argparse import ArgumentParser
from log_analyzer.helper_functions import create_model, train_model

"""
Entrypoint script for training
"""

def main(args):
    # Create the trainer+model
    trainer = create_model(args)
    # Train the model
    train_model(args, trainer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-folder", type=str, help="Path to data files.")
    parser.add_argument('--jagged', action='store_true',
                        help='Whether using sequences of variable length (Input should'
                             'be zero-padded to max_sequence_length.')
    parser.add_argument('--skipsos', action='store_true',
                        help='Whether to skip a start of sentence token.')
    parser.add_argument('--bidir', dest='bidirectional', action='store_true',
                        help='Whether to use bidirectional lstm for lower tier.')
    parser.add_argument('--tiered', action='store_true',
                        help='Whether to use tiered lstm model.')
    parser.add_argument('-bs', dest='batch_size', type=int, help='batch size.')
    parser.add_argument("-lstm_layers", nargs='+', type=int, default=[10],
                        help="A list of hidden layer sizes.")
    parser.add_argument("-context_layers", nargs='+', type=int, default=[10],
                        help="A list of context layer sizes.")
    parser.add_argument('-embed_dim', type=int, default=20,
                        help='Size of embeddings for categorical features.')
    parser.add_argument('--config', type=str, default='config.json', help='JSON configuration file')
    parser.add_argument('--model_dir', type=str, help='Directory to save stats and checkpoints to', default='runs')
    parser.add_argument('--load_from_checkpoint', type=str, help='Checkpoint to resume training from')
    args = parser.parse_args()
    main(args)