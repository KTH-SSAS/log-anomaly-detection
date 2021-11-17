import logging
import os
import socket
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import log_analyzer.application as application
import log_analyzer.data.data_loader as data_utils
import wandb
from log_analyzer.application import Application
from log_analyzer.config.model_config import LSTMConfig, ModelConfig, TieredLSTMConfig, TransformerConfig
from log_analyzer.config.trainer_config import DataConfig, TrainerConfig
from log_analyzer.tiered_trainer import TieredTrainer
from log_analyzer.trainer import LSTMTrainer, Trainer, TransformerTrainer

try:
    import torch
except ImportError:
    print("PyTorch is needed for this application.")

"""
Helper functions for model creation and training
"""

LSTM = "lstm"
TRANSFORMER = "transformer"
TIERED_LSTM = "tiered-lstm"


def calculate_max_input_length(data_length, bidirectional, skip_sos):
    """Maximum input length to model."""
    return data_length - 1 - int(skip_sos) + int(bidirectional)


def get_model_config(filename, model_type) -> ModelConfig:
    if model_type == TIERED_LSTM:
        return TieredLSTMConfig.init_from_file(filename)
    elif model_type == LSTM:
        return LSTMConfig.init_from_file(filename)
    elif model_type == TRANSFORMER:
        return TransformerConfig.init_from_file(filename)
    else:
        raise RuntimeError("Invalid model type.")


def create_identifier_string(model_name, comment=""):
    # TODO have model name be set by config, args or something else
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    id = f"{model_name}_{current_time}_{socket.gethostname()}_{comment}"
    return id


def init_from_args(args):
    return init_from_config_files(
        args.model_type,
        args.bidirectional,
        args.model_config,
        args.data_config,
        args.trainer_config,
        args.data_folder,
    )


def init_from_config_files(
    model_type: str,
    bidirectional,
    model_config_file: str,
    data_config_file: str,
    trainer_config_file: str,
    data_folder: str,
    base_logdir="runs",
):
    """Creates a model plus trainer given the specifications in args."""
    model_config = get_model_config(model_config_file, model_type)
    data_config = DataConfig.init_from_file(data_config_file)
    trainer_config = TrainerConfig.init_from_file(trainer_config_file)
    return init_from_config_classes(
        model_type,
        bidirectional,
        model_config,
        trainer_config,
        data_config,
        data_folder,
        base_logdir,
    )


def init_from_config_classes(
    model_type,
    bidirectional,
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    data_config: DataConfig,
    data_folder,
    base_logdir="runs",
):
    """Creates a model plus trainer given the specifications in args."""
    if not os.path.isdir(base_logdir):
        os.mkdir(base_logdir)
    id_string = create_identifier_string(model_type)
    log_dir = os.path.join(base_logdir, id_string)
    os.mkdir(log_dir)

    # Skip start of sequence token for forward models.
    skip_sos = not bidirectional

    shuffle_train_data = trainer_config.shuffle_train_data
    tokenization_type = data_config.tokenization
    if tokenization_type == "char":
        jagged = True
    elif tokenization_type == "word":
        jagged = False
    else:
        raise RuntimeError("Invalid tokenization.")

    # Settings for dataloader.

    max_input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)

    train_days = trainer_config.train_files
    test_days = trainer_config.test_files

    if data_config.tokenization == "word":
        if model_config.sequence_length is not None and model_config.sequence_length != max_input_length:
            raise RuntimeError(
                "Sequence length from model configuration does not match sequence length from data file."
            )
        else:
            model_config.sequence_length = max_input_length

    # Settings for model
    lm_trainer: Trainer
    if model_type == TIERED_LSTM and isinstance(model_config, TieredLSTMConfig):
        val_loader = None
        train_loader, test_loader = data_utils.load_data_tiered(
            data_folder,
            train_days,
            test_days,
            trainer_config.batch_size,
            bidirectional,
            skip_sos,
            jagged,
            max_input_length,
            num_steps=3,
            context_layers=model_config.context_layers,
        )
        lm_trainer = TieredTrainer(trainer_config, model_config, bidirectional, log_dir, train_loader)
    elif model_type == LSTM and isinstance(model_config, LSTMConfig):
        train_loader, val_loader, test_loader = data_utils.load_data(
            data_folder,
            train_days,
            test_days,
            trainer_config.batch_size,
            bidirectional,
            skip_sos,
            jagged,
            data_config.sentence_length,
            trainer_config.train_val_split,
            shuffle_train_data,
        )
        lm_trainer = LSTMTrainer(trainer_config, model_config, bidirectional, log_dir)
    elif model_type == TRANSFORMER and isinstance(model_config, TransformerConfig):
        train_loader, val_loader, test_loader = data_utils.load_data(
            data_folder,
            train_days,
            test_days,
            trainer_config.batch_size,
            bidirectional,
            skip_sos,
            jagged,
            data_config.sentence_length,
            trainer_config.train_val_split,
            shuffle_train_data,
        )
        lm_trainer = TransformerTrainer(trainer_config, model_config, log_dir)

    if Application.instance().wandb_initialized:
        wandb.config.update(model_config)
        wandb.config.update(data_config)
        wandb.config.update(trainer_config)

    Application.artifact_name = f"{model_type}-{data_config.tokenization}"
    Application.artifact_name += "-bidir" if bidirectional else ""

    return lm_trainer, train_loader, val_loader, test_loader

def validation_run(lm_trainer: Trainer, val_loader, val_run=0):
    # val_losses = []
    for val_iteration, val_batch in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            loss, *_ = lm_trainer.eval_step(val_batch, False)
            # val_losses.append(loss.item())
        if Application.instance().wandb_initialized:
            wandb.log(
                {
                    "valid/loss": loss,
                    "valid/iteration": val_iteration,
                    "valid/day": val_batch["day"][0],
                    "valid/run_number": val_run,
                }
            )
    return # val_losses

def train_model(lm_trainer: Trainer, train_loader, val_loader, test_loader, store_eval_data=True):
    """Perform 1 epoch of training on lm_trainer."""

    logger = logging.getLogger(application.TRAINER_LOGGER)

    outfile = None
    done = False
    log_dir = lm_trainer.checkpoint_dir
    writer = SummaryWriter(os.path.join(log_dir, "metrics"))

    if Application.instance().wandb_initialized:
        wandb.watch(lm_trainer.model)

    # True if val_loader is not None, False if val_loader is None
    run_validation = val_loader is not None
    if run_validation:
        # Number of times to run validation per epoch (including after each full epoch)
        validation_frequency = 10
        # Number of iterations between each validation run
        validation_period = (len(train_loader) // validation_frequency) + 1
        # Number of validation runs performed

    train_losses = []

    val_run = 0
    for iteration, batch in enumerate(tqdm(train_loader)):
        if isinstance(lm_trainer, TieredTrainer):
            if train_loader.flush is False:
                loss, done = lm_trainer.train_step(batch)
            else:
                logger.info(f"Due to flush, skipping the rest of the current file.")
                train_loader.skip_file = True
                continue
        else:
            loss, done = lm_trainer.train_step(batch)
        if Application.instance().wandb_initialized:
            wandb.log(
                {
                    "train/loss": loss,
                    "train/iteration": iteration,
                    "train/day": batch["day"][0],
                    "train/lr": lm_trainer.scheduler.get_last_lr()[0],
                }
            )
        train_losses.append(loss.item())
        writer.add_scalar(f'Loss/train_day_{batch["day"][0]}', loss, iteration)
        if run_validation and (iteration+1) % validation_period == 0:
            validation_run(lm_trainer, val_loader, val_run)
            val_run += 1

        if done:
            logger.info("Early stopping.")
            break

    if lm_trainer.config.early_stopping:
        lm_trainer.early_stopping.save_checkpoint()

    if run_validation:
        validation_run(lm_trainer, val_loader, val_run)
        val_run += 1

    test_losses = []
    for iteration, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            loss, *_ = lm_trainer.eval_step(batch, store_eval_data)
            test_losses.append(loss.item())
        writer.add_scalar(f'Loss/test_day_{batch["day"][0]}', loss, iteration)
        if Application.instance().wandb_initialized:
            wandb.log(
                {
                    "eval/loss": loss,
                    "eval/iteration": iteration,
                    "eval/day": batch["day"][0],
                }
            )
        if outfile is not None:
            for line, sec, day, usr, red, loss in zip(
                batch["line"].flatten().tolist(),
                batch["second"].flatten().tolist(),
                batch["day"].flatten().tolist(),
                batch["user"].flatten().tolist(),
                batch["red"].flatten().tolist(),
                loss.flatten().tolist(),
            ):
                outfile.write("%s %s %s %s %s %s %r\n" % (iteration, line, sec, day, usr, red, loss))

    writer.close()

    model_save_path = os.path.join(log_dir, "model.pt")
    torch.save(lm_trainer.model, model_save_path)

    if Application.instance().wandb_initialized:
        # Save the model weights as a versioned artifact
        artifact = wandb.Artifact(
            Application.artifact_name,
            "model",
            metadata=lm_trainer.model.config.__dict__,
        )
        artifact.add_file(model_save_path)
        artifact.save()

    lm_trainer.config.save_config(os.path.join(log_dir, "trainer_config.json"))
    lm_trainer.model.config.save_config(os.path.join(log_dir, "model_config.json"))
    return train_losses, test_losses
