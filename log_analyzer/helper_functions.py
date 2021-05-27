from argparse import ArgumentParser, Namespace
from log_analyzer.model.lstm import Tiered_LSTM
from log_analyzer.config.config import Config

from torch.utils.data.dataset import ConcatDataset
from log_analyzer.config.model_config import LSTMConfig, TieredLSTMConfig
from log_analyzer.config.trainer_config import DataConfig, TrainerConfig
import os
import json
import socket
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import log_analyzer.data.data_loader as data_utils
from log_analyzer.trainer import LSTMTrainer, Trainer
from log_analyzer.tiered_trainer import TieredTrainer
try:
    import torch
except ImportError:
    print('PyTorch is needed for this application.')

"""
Helper functions for model creation and training
"""

LSTM = 'lstm'
TRANSFORMER = 'transformer'
TIERED_LSTM = 'tiered-lstm' 

def generate_trainer_config(args : Namespace):
    """Generate configs based on args and conf file. Intermediary function while refactoring"""

    data_config = DataConfig(train_files, test_files=conf['test_files'], sentence_length=conf['sentence_length'], 
    vocab_size=conf['token_set_size'], number_of_days=conf['num_days'])

    trainer_config : TrainerConfig = TrainerConfig(data_config.__dict__,
        batch_size=batch_size, jagged=jagged, bidirectional=bidirectional,
        tiered=tiered, learning_rate=conf['lr'], early_stopping=True,
        early_stop_patience=conf['patience'], scheduler_gamma=conf['gamma'],
        scheduler_step_size=conf['step_size'])

    return trainer_config, model_config

def get_model_config(filename, model_type) -> Config:
    if model_type == TIERED_LSTM:
        return TieredLSTMConfig.init_from_file(filename)
    elif model_type == LSTM:
        return LSTMConfig.init_from_file(filename)
    elif model_type == TRANSFORMER:
        raise NotImplementedError("Transformer not yet implemented.")
    else:
        raise RuntimeError('Invalid model type.')

def create_identifier_string(model_name, comment=""):
    #TODO have model name be set by config, args or something else
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    id = f'{model_name}_{current_time}_{socket.gethostname()}_{comment}'
    return id


def create_model_args(args):
    return create_model(args.model_type, args.model_config, args.trainer_config, args.data_folder, 
    args.bidirectional, args.skipsos, args.jagged)

def create_model(model_type, model_config_file, trainer_config_file, data_folder, bidirectional, skipsos, jagged):
    """Creates a model plus trainer given the specifications in args"""
    base_logdir = 'runs'
    if not os.path.isdir(base_logdir):
        os.mkdir(base_logdir)
    id_string = create_identifier_string(model_type)
    log_dir = os.path.join(base_logdir, id_string)
    os.mkdir(log_dir)

    model_config : Config = get_model_config(model_config_file, model_type)
    trainer_config = TrainerConfig.init_from_file(trainer_config_file)
    bidir = bidirectional
    trainer_config.bidirectional = bidir
    model_config.bidirectional = bidir
    trainer_config.jagged = jagged
    model_config.jagged = jagged

    verbose = True

    # Settings for dataloader.
    
    data_config = trainer_config.data_config
    max_input_length = data_config.sentence_length - 1 - int(skipsos) + int(bidir)
    train_days = data_config.train_files
    test_days = data_config.test_files

    # Settings for LSTM.
    if model_type == TIERED_LSTM:
        model_config: TieredLSTMConfig = model_config
        train_loader, test_loader = data_utils.load_data_tiered(data_folder, train_days, test_days,
        trainer_config.batch_size, bidir, skipsos, jagged, max_input_length, num_steps=3, context_layers=model_config.context_layers)
        lm_trainer = TieredTrainer(trainer_config, model_config, log_dir, verbose, train_loader)
    else:
        train_loader, test_loader = data_utils.load_data(data_folder, train_days, test_days,
        trainer_config.batch_size, bidir, skipsos, jagged, max_input_length)
        lm_trainer = LSTMTrainer(trainer_config, model_config, log_dir, verbose)

    return lm_trainer, train_loader, test_loader


def train_model(lm_trainer : Trainer, train_loader, test_loader):
    """Perform 1 epoch of training on lm_trainer"""

    outfile = None
    verbose = False
    log_dir = lm_trainer.checkpoint_dir
    writer = SummaryWriter(os.path.join(log_dir, 'metrics'))

    train_losses = []
    for iteration, batch in enumerate(train_loader):
        if lm_trainer is TieredTrainer:
            if train_loader.flush is False:
                loss, done = lm_trainer.train_step(batch)
            else:
                loss, *_ = lm_trainer.eval_step(batch)
                print(f'Due to flush, training stopped... Current loss: {loss:.3f}')
        else: 
            loss, done = lm_trainer.train_step(batch)
        train_losses.append(loss.item())
        writer.add_scalar(f'Loss/train_day_{batch["day"][0]}', loss, iteration)
        if done:
            print("Early stopping.")
            break

    test_losses = []
    for iteration, batch in enumerate(test_loader):
        loss, *_ = lm_trainer.eval_step(batch)
        test_losses.append(loss.item())
        writer.add_scalar(f'Loss/test_day_{batch["day"][0]}', loss, iteration)

        if outfile is not None:
            for line, sec, day, usr, red, loss in zip(batch['line'].flatten().tolist(),
                                                    batch['second'].flatten().tolist(),
                                                    batch['day'].flatten().tolist(),
                                                    batch['user'].flatten().tolist(),
                                                    batch['red'].flatten().tolist(),
                                                    loss.flatten().tolist()):
                outfile.write('%s %s %s %s %s %s %r\n' % (iteration, line, sec, day, usr, red, loss))

        if verbose:
            print(f"{batch['x'].shape[0]}, {batch['line'][0]}, {batch['second'][0]} fixed {batch['day']} {loss}") 
            # TODO: I don't think this print line, but I decided to keep it since removing a line is always easier than adding a line.
            #       Also, In the original code, there was {data.index} which seems to be an accumulated sum of batch sizes. 
            #       I don't think we need {data.index}. but... I added it to to-do since we might need to do it in future.
    
    writer.close()
    torch.save(lm_trainer.model, os.path.join(log_dir,'model.pt'))
    lm_trainer.config.save_config(os.path.join(log_dir, 'trainer_config.json'))
    lm_trainer.model.config.save_config(os.path.join(log_dir, 'model_config.json'))
    return train_losses, test_losses
