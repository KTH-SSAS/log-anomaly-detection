"""Utility script to convert from the old args and config files to new versions.
    Will produce a split trainer/model config file in the same directory as the given config file
"""
from argparse import ArgumentParser, Namespace
from log_analyzer.config.model_config import LSTMConfig, TieredLSTMConfig
from log_analyzer.config.trainer_config import TrainerConfig, DataConfig
import json
import os.path

def convert_configs(args : Namespace, conf : dict):
    """Generate configs based on args and conf file. Intermediary function while refactoring"""

    data_config = DataConfig(conf['train_files'], test_files=conf['test_files'], sentence_length=conf['sentence_length'], 
    vocab_size=conf['token_set_size'], number_of_days=conf['num_days'])

    trainer_config : TrainerConfig = TrainerConfig(data_config.__dict__,
        batch_size=args.batch_size, jagged=args.jagged, bidirectional=args.bidirectional,
        tiered=args.tiered, learning_rate=conf['lr'], early_stopping=True,
        early_stop_patience=conf['patience'], scheduler_gamma=conf['gamma'],
        scheduler_step_size=conf['step_size'])

    if args.tiered:
        model_config = TieredLSTMConfig(args.lstm_layers, conf['token_set_size'], args.embed_dim, args.bidirectional, None, 0, args.jagged, args.context_layers)
    else:
        model_config = LSTMConfig(args.lstm_layers, conf['token_set_size'], args.embed_dim, args.bidirectional, None, 0, args.jagged)
    return trainer_config, model_config

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
    arguments = parser.parse_args()
      
    with open(arguments.config, 'r') as f:
        config = json.load(f)
    
    trainer_config, model_config = convert_configs(arguments, config)

    conf_file_name = os.path.splitext(os.path.basename(arguments.config))[0]
    dirname = os.path.split(arguments.config)[0]
    trainer_config.save_config(os.path.join(dirname, f"{conf_file_name}_trainer.json"))
    model_config.save_config(os.path.join(dirname, f"{conf_file_name}_model.json"))
