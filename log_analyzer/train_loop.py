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
from tqdm import tqdm
import wandb
import log_analyzer.application as application

import logging

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

def calculate_max_input_length(data_length, bidirectional, skip_sos):
    """
    Maximum input length to model.
    """
    return data_length - 1 - int(skip_sos) + int(bidirectional)


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
    # TODO have model name be set by config, args or something else
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    id = f'{model_name}_{current_time}_{socket.gethostname()}_{comment}'
    return id


def init_from_args(args):
    return init_from_config_files(
        args.model_type, args.bidirectional,
        args.model_config, args.data_config,
        args.trainer_config, args.data_folder)

def init_from_config_classes(model_type, bidirectional, model_config: LSTMConfig, trainer_config: TrainerConfig, data_config: DataConfig, data_folder, base_logdir='runs'):
    """Creates a model plus trainer given the specifications in args"""
    if not os.path.isdir(base_logdir):
        os.mkdir(base_logdir)
    id_string = create_identifier_string(model_type)
    log_dir = os.path.join(base_logdir, id_string)
    os.mkdir(log_dir)

    skip_sos = not bidirectional #Skip start of sequence token for forward models.

    tokenization_type = data_config.tokenization
    if tokenization_type == 'char':
        jagged = True
    elif tokenization_type == 'word':
        jagged = False
    else:
        raise RuntimeError("Invalid tokenization.")

    # Settings for dataloader.

    max_input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)

    train_days = trainer_config.train_files
    test_days = trainer_config.test_files

    if data_config.tokenization == 'word':
        if model_config.sequence_length is not None and model_config.sequence_length != max_input_length:
            raise RuntimeError('Sequence length from model configuration does not match sequence length from data file.')
        else:
            model_config.sequence_length = max_input_length

    # Settings for LSTM.
    if model_type == TIERED_LSTM:
        model_config: TieredLSTMConfig = model_config
        train_loader, test_loader = data_utils.load_data_tiered(data_folder, train_days, test_days,
                                                                trainer_config.batch_size, bidirectional, skip_sos, jagged,
                                                                max_input_length, num_steps=3,
                                                                context_layers=model_config.context_layers)
        lm_trainer = TieredTrainer(
            trainer_config, model_config, bidirectional, log_dir, train_loader)
    else:
        train_loader, test_loader = data_utils.load_data(data_folder, train_days, test_days,
                                                         trainer_config.batch_size, bidirectional, skip_sos, jagged,
                                                         max_input_length)
        lm_trainer = LSTMTrainer(
            trainer_config, model_config, bidirectional, log_dir)

    if application.wandb_initalized:
        wandb.config.update(model_config)
        wandb.config.update(data_config)
        wandb.config.update(trainer_config)
    
    application.artifact_name = f"{model_type}-{data_config.tokenization}"
    application.artifact_name += "-bidir" if bidirectional else ""

    return lm_trainer, train_loader, test_loader


def init_from_config_files(model_type: str, bidirectional, model_config_file: str, data_config_file: str, trainer_config_file: str, data_folder: str, base_logdir='runs'):
    """Creates a model plus trainer given the specifications in args"""
    model_config = get_model_config(model_config_file, model_type)
    data_config = DataConfig.init_from_file(data_config_file)
    trainer_config = TrainerConfig.init_from_file(trainer_config_file)
    return init_from_config_classes(model_type, bidirectional, model_config, trainer_config, data_config, data_folder, base_logdir)


def train_model(lm_trainer: Trainer, train_loader, test_loader, store_eval_data=False):
    """Perform 1 epoch of training on lm_trainer"""

    logger = logging.getLogger(application.TRAINER_LOGGER)

    outfile = None
    done = False
    log_dir = lm_trainer.checkpoint_dir
    writer = SummaryWriter(os.path.join(log_dir, 'metrics'))

    if application.wandb_initalized:
        wandb.watch(lm_trainer.model)

    train_losses = []
    for iteration, batch in enumerate(tqdm(train_loader)):
        if type(lm_trainer) is TieredTrainer:
            if train_loader.flush is False:
                loss, done = lm_trainer.train_step(batch)
            else:
                logger.info(f'Due to flush, skipping the rest of the current file.')
                train_loader.skip_file = True
                continue
        else:
            loss, done = lm_trainer.train_step(batch)
            if application.wandb_initalized:
                wandb.log({"train/loss": loss, "train/iteration": iteration, "train/day": batch["day"][0]})
        train_losses.append(loss.item())
        writer.add_scalar(f'Loss/train_day_{batch["day"][0]}', loss, iteration)
        if done:
            logger.info("Early stopping.")
            break

    lm_trainer.early_stopping.save_checkpoint()
    
    test_losses = []
    for iteration, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            loss, *_ = lm_trainer.eval_step(batch, store_eval_data)
            test_losses.append(loss.item())
        writer.add_scalar(f'Loss/test_day_{batch["day"][0]}', loss, iteration)
        if application.wandb_initalized:
            wandb.log({"eval/loss": loss, "eval/iteration": iteration, "eval/day": batch["day"][0]})
        if outfile is not None:
            for line, sec, day, usr, red, loss in zip(batch['line'].flatten().tolist(),
                                                      batch['second'].flatten().tolist(),
                                                      batch['day'].flatten().tolist(),
                                                      batch['user'].flatten().tolist(),
                                                      batch['red'].flatten().tolist(),
                                                      loss.flatten().tolist()):
                outfile.write('%s %s %s %s %s %s %r\n' %
                              (iteration, line, sec, day, usr, red, loss))

    writer.close()
    
    model_save_path = os.path.join(log_dir, 'model.pt')
    torch.save(lm_trainer.model, model_save_path)  

    if application.wandb_initalized:
        # Save the model weights as a versioned artifact
        artifact = wandb.Artifact(application.artifact_name, "model", metadata=lm_trainer.model.config.__dict__)
        artifact.add_file(model_save_path)
        artifact.save()

    lm_trainer.config.save_config(os.path.join(log_dir, 'trainer_config.json'))
    lm_trainer.model.config.save_config(
        os.path.join(log_dir, 'model_config.json'))
    return train_losses, test_losses
