import logging
import os
import socket
from datetime import datetime

import numpy as np
from tqdm import tqdm

import log_analyzer.application as application
import log_analyzer.data.data_loader as data_utils
import wandb
from log_analyzer.application import Application
from log_analyzer.config.model_config import (
    LSTMConfig,
    ModelConfig,
    TieredLSTMConfig,
    TieredTransformerConfig,
    TransformerConfig,
)
from log_analyzer.config.trainer_config import DataConfig, TrainerConfig
from log_analyzer.evaluator import Evaluator
from log_analyzer.model.lstm import BidLSTM, FwdLSTM, LogModel, TieredLSTM
from log_analyzer.model.transformer import TieredTransformer, Transformer
from log_analyzer.trainer import Trainer

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
TIERED_TRANSFORMER = "tiered-transformer"

LOGGING_FREQUENCY = 10  # How often to log results. Set to 1 to log everything.
VALIDATION_FREQUENCY = 10  # Number of times to do validation per epoch. Set to 1 to only validate after each epoch.


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
    elif model_type == TIERED_TRANSFORMER:
        return TieredTransformerConfig.init_from_file(filename)
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

    if model_type in (TIERED_LSTM, TIERED_TRANSFORMER):
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
        )
    elif model_type in (LSTM, TRANSFORMER):
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
    else:
        raise RuntimeError("Invalid model type.")

    # Settings for model
    model = init_model(model_config, bidirectional)

    # Trainer
    lm_trainer = Trainer(trainer_config, model, log_dir)

    # Evaluator
    lm_evaluator = Evaluator(model, log_dir)

    if Application.instance().wandb_initialized:
        wandb.config.update(model_config)
        wandb.config.update(data_config)
        wandb.config.update(trainer_config)

    Application.artifact_name = f"{model_type}-{data_config.tokenization}"
    Application.artifact_name += "-bidir" if bidirectional else ""

    return lm_trainer, lm_evaluator, train_loader, val_loader, test_loader


def init_model(model_config: ModelConfig, bidirectional) -> LogModel:
    """Initialises a new model based on the model config."""
    if type(model_config) == TieredLSTMConfig:
        # TieredLSTMConfig is a type of LSTMConfig, so check for tiered first
        return TieredLSTM(model_config, bidirectional)
    elif type(model_config) == LSTMConfig:
        model = BidLSTM(model_config) if bidirectional else FwdLSTM(model_config)
        return model
    elif type(model_config) == TieredTransformerConfig:
        # TieredTransformerConfig is a type of TransformerConfig, so check for tiered first
        return TieredTransformer(model_config)
    elif type(model_config) == TransformerConfig:
        return Transformer(model_config)
    else:
        raise RuntimeError("Invalid model config type.")


def wandb_log(iteration, frequency, data: dict):
    # Don't log every result (unless LOGGING_FREQUENCY is 1)
    if iteration % frequency == 0:
        if Application.instance().wandb_initialized:
            wandb.log(data)


def train_model(lm_trainer: Trainer, train_loader, val_loader):
    """Perform training on lm_trainer."""
    logger = logging.getLogger(application.TRAINER_LOGGER)

    def validation_run(train_iteration=0, val_run=0):
        """Performs one phase of validation on lm_trainer."""
        val_losses = []
        for val_iteration, val_batch in enumerate(tqdm(val_loader, desc=f"Valid:{val_run:2d}")):
            split_batch = val_loader.split_batch(val_batch)
            with torch.no_grad():
                loss, *_ = lm_trainer.train_step(split_batch, validation=True)
                val_losses.append(loss.item())
            # Log the current validation loss and val_iteration to enable detailed view of
            # validation loss.
            # Also log the current train iteration and validation run_number to enable
            # overview analysis of each validation run
            wandb_log(
                val_iteration,
                LOGGING_FREQUENCY,
                {
                    "valid/loss": loss,
                    "valid/run_number": val_run,
                    "valid/iteration": val_iteration,
                    "train/iteration": train_iteration,
                },
            )
        mean_val_loss = np.mean(val_losses)
        lm_trainer.early_stopping(mean_val_loss)

    done = False
    log_dir = lm_trainer.checkpoint_dir
    epochs = lm_trainer.config.epochs

    if Application.instance().wandb_initialized:
        wandb.watch(lm_trainer.model)

    # True if val_loader is not None, False if val_loader is None
    run_validation = val_loader is not None
    if run_validation:
        # Number of iterations between each validation run
        validation_period = (len(train_loader) // VALIDATION_FREQUENCY) + 1

    train_losses = []

    val_run = 0
    iteration = 0
    for epoch in tqdm(range(epochs), desc="Epoch   "):
        # Shuffle train data order for each epoch?
        # Count iteration continuously up through each epoch
        for epoch_iteration, batch in enumerate(tqdm(train_loader, desc="Training")):
            # epoch_iteration = iterations in this epoch (used to determine when to run validation)
            iteration += 1  # Total iterations in training (cumulative)
            # Split the batch
            split_batch = train_loader.split_batch(batch)
            if lm_trainer.model.tiered:
                if train_loader.flush is False:
                    loss, done = lm_trainer.train_step(split_batch)
                else:
                    logger.info(f"Due to flush, skipping the rest of the current file.")
                    train_loader.skip_file = True
                    continue
            else:
                loss, done = lm_trainer.train_step(split_batch)
            train_losses.append(loss.item())
            wandb_log(
                epoch_iteration,
                LOGGING_FREQUENCY,
                {
                    "train/loss": loss,
                    "train/iteration": iteration,
                    "train/day": batch["day"][0],
                    "train/lr": lm_trainer.scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                },
            )
            if run_validation and epoch_iteration > 0 and (epoch_iteration % validation_period == 0):
                validation_run(iteration, val_run)
                val_run += 1

            if done:
                logger.info("Early stopping.")
                break

        if run_validation:
            validation_run(iteration, val_run)
            val_run += 1

        if done:
            break

    if lm_trainer.config.early_stopping:
        # Save the best performing model version to file
        lm_trainer._EarlyStopping.save_checkpoint()

    # Save the final model version to file
    model_save_path = os.path.join(log_dir, "model.pt")
    torch.save(lm_trainer.model.state_dict(), model_save_path)

    lm_trainer.config.save_config(os.path.join(log_dir, "trainer_config.json"))
    lm_trainer.model.config.save_config(os.path.join(log_dir, "model_config.json"))
    return train_losses


def eval_model(lm_evaluator: Evaluator, test_loader, store_eval_data=False, model_file_name="model.pt"):
    """Perform testing on lm_trainer.

    Note: model_file_name is only used for uploading model parameters to wandb.
    """
    log_dir = lm_evaluator.checkpoint_dir
    model_save_path = os.path.join(log_dir, model_file_name)

    if Application.instance().wandb_initialized:
        # Save the model weights as a versioned artifact
        artifact = wandb.Artifact(
            Application.artifact_name,
            "model",
            metadata=lm_evaluator.model.config.__dict__,
        )
        artifact.add_file(model_save_path)
        artifact.save()

    test_losses = []
    for iteration, batch in enumerate(tqdm(test_loader, desc="Test")):
        split_batch = test_loader.split_batch(batch)
        with torch.no_grad():
            loss, *_ = lm_evaluator.eval_step(split_batch, store_eval_data=store_eval_data)
            test_losses.append(loss.item())
            wandb_log(
                iteration,
                LOGGING_FREQUENCY,
                {
                    "eval/loss": loss,
                    "eval/iteration": iteration,
                    "eval/day": batch["day"][0],
                },
            )

    return test_losses
