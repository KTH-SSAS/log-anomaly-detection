"""Helper functions for model creation and training."""
import json
import logging
import os
import socket
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import log_analyzer.data.data_loader as data_utils
import wandb
from log_analyzer import application
from log_analyzer.application import Application
from log_analyzer.config import TrainerConfig
from log_analyzer.config.model_config import (
    LSTMConfig,
    ModelConfig,
    MultilineTransformerConfig,
    TieredLSTMConfig,
    TieredTransformerConfig,
    TransformerConfig,
)
from log_analyzer.evaluator import Evaluator
from log_analyzer.model.lstm import BidLSTM, FwdLSTM, LogModel, TieredLSTM
from log_analyzer.model.model_util import DelayedKeyboardInterrupt
from log_analyzer.model.transformer import MultilineTransformer, TieredTransformer, Transformer
from log_analyzer.tokenizer.tokenizer_neo import (
    CharTokenizer,
    GlobalTokenizer,
    GlobalVocab,
    LANLTokenizer,
    LANLVocab,
    Tokenizer,
)
from log_analyzer.tokenizer.vocab import MergedLANLVocab
from log_analyzer.trainer import Trainer

try:
    import torch
except ImportError:
    print("PyTorch is needed for this application.")


LSTM: str = "lstm"
TRANSFORMER: str = "transformer"
TIERED_LSTM: str = "tiered-lstm"
TIERED_TRANSFORMER: str = "tiered-transformer"
MULTILINE_TRANSFORMER: str = "multiline-transformer"

LOGGING_FREQUENCY: int = 10  # How often to log results. Set to 1 to log everything.

# Autosave every 10 minutes
AUTOSAVE_TIME = 10 * 60

WORD_GLOBAL = "word-global"
WORD_FIELDS = "word-fields"
WORD_MERGED = "word-merged"
CHAR = "char"

tokenizer_vocabs = {
    CHAR: (CharTokenizer, None),
    WORD_FIELDS: (LANLTokenizer, LANLVocab),
    WORD_GLOBAL: (GlobalTokenizer, GlobalVocab),
    WORD_MERGED: (LANLTokenizer, MergedLANLVocab),
}


def get_tokenizer(tokenization, counts_file: Path, cutoff) -> Tokenizer:
    tokenizer: Tokenizer
    vocab = None
    tokenizer_cls, vocab_cls = tokenizer_vocabs[tokenization]
    if counts_file is not None:
        with open(counts_file, encoding="utf8") as f:
            counts = json.load(f)
        users = list(counts["src_user"].keys())
        vocab = vocab_cls.counts2vocab(counts, cutoff) if vocab_cls is not None else None
    else:
        users = None

    tokenizer = tokenizer_cls(vocab, users)
    return tokenizer


def get_task(model: str, bidirectional: bool) -> str:
    """Return the language modeling task for the given model, since it varies
    depending on its directionality."""
    if bidirectional and model in (TRANSFORMER, TIERED_TRANSFORMER):
        return data_utils.MASKED_LM
    if bidirectional and model in (LSTM, TIERED_LSTM):
        return data_utils.BIDIR_LSTM_LM
    if model == MULTILINE_TRANSFORMER:
        return data_utils.SENTENCE_LM

    return data_utils.AUTOREGRESSIVE_LM


def calculate_max_input_length(task: str, tokenizer: Tokenizer) -> Optional[int]:
    """Maximum input length to model."""
    add_sos, add_eos = data_utils.tokens_to_add(task)
    seq_len = tokenizer.sequence_length
    if seq_len is None:
        return None
    seq_len -= 1 if task == data_utils.AUTOREGRESSIVE_LM else 0
    return int(add_sos) + seq_len + int(add_eos)


def get_model_config(filename: Path, model_type: str) -> ModelConfig:
    if model_type == TIERED_LSTM:
        return TieredLSTMConfig.init_from_file(filename)
    if model_type == LSTM:
        return LSTMConfig.init_from_file(filename)
    if model_type == TRANSFORMER:
        return TransformerConfig.init_from_file(filename)
    if model_type == TIERED_TRANSFORMER:
        return TieredTransformerConfig.init_from_file(filename)
    if model_type == MULTILINE_TRANSFORMER:
        return MultilineTransformerConfig.init_from_file(filename)

    raise RuntimeError("Invalid model type.")


def create_identifier_string(model_name: str, tokenization: str) -> str:
    if Application.instance().wandb_initialized:
        id_string = f"{wandb.run.id}_{model_name}_{tokenization}"
    else:
        current_time = datetime.now().strftime(r"%m-%d_%H:%M:%S")
        id_string = f"{model_name}_{tokenization}_{current_time}@{socket.gethostname()}"
    return id_string


def init_from_args(args: Namespace) -> Tuple[Trainer, Evaluator, DataLoader, DataLoader, DataLoader]:
    return init_from_config_files(
        args.model_type,
        args.bidirectional,
        Path(args.model_config),
        args.tokenization,
        Path(args.trainer_config),
        Path(args.data_folder),
        counts_file=Path(args.counts_file),
    )


def init_from_config_files(
    model_type: str,
    bidirectional: bool,
    model_config_file: Path,
    tokenization: str,
    trainer_config_file: Path,
    data_folder: Path,
    base_logdir=Path("./runs"),
    counts_file=None,
) -> Tuple[Trainer, Evaluator, DataLoader, DataLoader, DataLoader]:

    """Creates a model plus trainer given the specifications in args."""
    model_config = get_model_config(model_config_file, model_type)
    trainer_config = TrainerConfig.init_from_file(trainer_config_file)
    return init_from_config_classes(
        model_type,
        bidirectional,
        model_config,
        trainer_config,
        tokenization,
        data_folder,
        base_logdir,
        counts_file=counts_file,
    )


def init_from_config_classes(
    model_type,
    bidirectional,
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    tokenization: str,
    data_folder,
    base_logdir: Path = Path("./runs"),
    counts_file=None,
    cutoff=40,
):
    """Creates a model plus trainer given the specifications in args."""
    if not base_logdir.is_dir():
        os.mkdir(base_logdir)
    id_string = create_identifier_string(model_type, tokenization)
    log_dir: Path = base_logdir / id_string
    log_dir.mkdir()

    shuffle_train_data = trainer_config.shuffle_train_data

    tokenizer = get_tokenizer(tokenization, counts_file, cutoff)

    task = get_task(model_type, bidirectional)

    train_files = trainer_config.train_files
    validation_files = trainer_config.validation_files
    test_files = trainer_config.test_files

    model_config.vocab_size = tokenizer.vocab_size
    model_config.sequence_length = calculate_max_input_length(task, tokenizer)

    if isinstance(model_config, TieredTransformerConfig):
        model_config.number_of_users = tokenizer.num_users

    if model_type in (TIERED_LSTM, TIERED_TRANSFORMER):
        train_loader, val_loader, test_loader = data_utils.load_data_tiered(
            data_folder,
            train_files,
            validation_files,
            test_files,
            (trainer_config.train_batch_size, trainer_config.eval_batch_size),
            tokenizer,
            task,
            num_steps=3,
        )
    elif model_type in (LSTM, TRANSFORMER):
        train_loader, val_loader, test_loader = data_utils.load_data(
            data_folder,
            train_files,
            validation_files,
            test_files,
            (trainer_config.train_batch_size, trainer_config.eval_batch_size),
            tokenizer,
            task,
            shuffle_train_data,
        )
    elif model_type in (MULTILINE_TRANSFORMER) and isinstance(model_config, MultilineTransformerConfig):
        train_loader, val_loader, test_loader = data_utils.load_data_multiline(
            data_folder,
            train_files,
            validation_files,
            test_files,
            (trainer_config.train_batch_size, trainer_config.eval_batch_size),
            tokenizer,
            task,
            model_config.shift_window,
            model_config.memory_type,
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
        wandb.config.update(trainer_config)

    Application.artifact_name = f"{model_type}-{tokenization}"
    Application.artifact_name += "-bidir" if bidirectional else ""

    return lm_trainer, lm_evaluator, train_loader, val_loader, test_loader


def init_model(model_config: ModelConfig, bidirectional) -> LogModel:
    """Initialises a new model based on the model config."""
    if isinstance(model_config, TieredLSTMConfig):
        # TieredLSTMConfig is a type of LSTMConfig, so check for tiered first
        return TieredLSTM(model_config, bidirectional)
    if isinstance(model_config, LSTMConfig):
        model = BidLSTM(model_config) if bidirectional else FwdLSTM(model_config)
        return model
    if isinstance(model_config, TieredTransformerConfig):
        # TieredTransformerConfig is a type of TransformerConfig, so check for tiered first
        return TieredTransformer(model_config, bidirectional)
    if isinstance(model_config, MultilineTransformerConfig):
        # MultilineTransformerConfig is a type of TransformerConfig, so check for Multiline first
        return MultilineTransformer(model_config, bidirectional)
    if isinstance(model_config, TransformerConfig):
        return Transformer(model_config, bidirectional)

    raise RuntimeError("Invalid model config type.")


def wandb_log(iteration, frequency, data: dict):
    # Don't log every result (unless LOGGING_FREQUENCY is 1)
    if Application.instance().wandb_initialized:
        if iteration % frequency == 0:
            wandb.log(data)


def train_model(lm_trainer: Trainer, train_loader, val_loader):
    """Perform training on lm_trainer."""
    logger = logging.getLogger(application.TRAINER_LOGGER)
    last_save = time()

    best_val_score = 1000000

    @torch.inference_mode()
    def validation_run(train_iteration=0, val_run=0, best_score=1e6):
        """Performs one phase of validation on lm_trainer."""
        if (
            isinstance(val_loader.dataset, (data_utils.IterableLogDataset, data_utils.IterableUserMultilineDataset))
            and epoch > 0
        ):
            # Refresh the iterator so we can run another epoch
            val_loader.dataset.refresh_iterator()
        val_losses = []
        val_iteration = 0
        for val_iteration, val_batch in enumerate(tqdm(val_loader, desc=f"Valid:{val_run:2d}")):
            # Only allow interrupt between each batch
            with DelayedKeyboardInterrupt():
                split_batch = val_loader.split_batch(val_batch)
                # Check that the split batch contains entries (see MultilineDataloader's mask filtering)
                if len(split_batch["X"]) == 0:
                    continue
                loss, *_ = lm_trainer.train_step(split_batch, validation=True)
                val_losses.append(loss.item())
        mean_val_loss = np.mean(val_losses)
        # Log the mean validation loss and the current train iteration and validation run_number
        wandb_log(
            val_iteration,
            1,
            {
                "valid/loss": mean_val_loss,
                "valid/run_number": val_run,
                "train/iteration": train_iteration,
            },
        )

        if mean_val_loss < best_score:
            model_save_path = log_dir / "model_best.pt"
            torch.save(lm_trainer.model.state_dict(), model_save_path)
            new_best = mean_val_loss
        else:
            new_best = best_score

        lm_trainer.early_stopping(mean_val_loss)
        return new_best

    done = False
    log_dir = lm_trainer.checkpoint_dir
    epochs = lm_trainer.config.epochs

    if Application.instance().wandb_initialized:
        wandb.watch(lm_trainer.model)

    # True if val_loader is not None, False if val_loader is None
    run_validation = val_loader is not None
    if run_validation and lm_trainer.config.validations_per_epoch > 0:
        # Number of iterations between each validation run
        validation_period = (len(train_loader) // lm_trainer.config.validations_per_epoch) + 1
    else:
        validation_period = 0

    train_losses = []

    val_run = 0
    iteration = 0
    try:
        for epoch in tqdm(range(epochs), desc="Epoch   "):
            if (
                isinstance(
                    train_loader.dataset, (data_utils.IterableLogDataset, data_utils.IterableUserMultilineDataset)
                )
                and epoch > 0
            ):
                # Refresh the iterator so we can run another epoch
                train_loader.dataset.refresh_iterator()
            # Shuffle train data order for each epoch?
            # Count iteration continuously up through each epoch
            for epoch_iteration, batch in enumerate(tqdm(train_loader, desc="Training")):
                # Only allow interrupt between each batch
                with DelayedKeyboardInterrupt():
                    # epoch_iteration = iterations in this epoch (used to determine when to run validation)
                    # Split the batch
                    split_batch = train_loader.split_batch(batch)
                    if lm_trainer.model.tiered:
                        if train_loader.flush is False:
                            loss, gradient_norm, done = lm_trainer.train_step(split_batch)
                        else:
                            if iteration == 0:
                                raise Exception("Flush happened before any training could be done.")

                            logger.info("Due to flush, skipping the rest of the current file.")
                            train_loader.skip_file = True
                            continue
                    else:
                        loss, gradient_norm, done = lm_trainer.train_step(split_batch)

                    if (time() - last_save) > AUTOSAVE_TIME:
                        tqdm.write("Autosaving...")
                        last_save = time()
                        with open(log_dir / "autosave.pt", "wb") as f:
                            torch.save(lm_trainer.model.state_dict(), f)

                    iteration += 1  # Total iterations in training (cumulative)
                    train_losses.append(loss.item())
                    wandb_log(
                        epoch_iteration,
                        LOGGING_FREQUENCY,
                        {
                            "train/loss": loss,
                            "train/iteration": iteration,
                            "train/day": batch["day"][0],
                            "train/lr": lm_trainer.optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                            "train/gradient_norm": gradient_norm,
                        },
                    )
                # Validation within an epoch
                if validation_period > 0 and epoch_iteration > 0 and (epoch_iteration % validation_period == 0):
                    best_val_score = validation_run(iteration, val_run, best_val_score)
                    val_run += 1

            if lm_trainer.epoch_scheduler is not None:
                lm_trainer.epoch_scheduler.step()

            # Validation after epoch
            if run_validation:
                best_val_score = validation_run(iteration, val_run, best_val_score)
                val_run += 1

            if done:
                break
    except KeyboardInterrupt:
        print("Ctrl+C received, cancelling training and exiting.")
        # Save the current model version
        lm_trainer.earlystopping.store_state_dict(np.Inf, lm_trainer.model)
        lm_trainer.earlystopping.save_checkpoint()
        # exit
        sys.exit(f"Training interrupted, most recent model has been saved to '{lm_trainer.earlystopping.path}'.")

    if lm_trainer.config.early_stopping:
        # Save the best performing model version to file
        lm_trainer.earlystopping.save_checkpoint()

    # Save the final model version to file
    model_save_path = log_dir / "model.pt"
    torch.save(lm_trainer.model.state_dict(), model_save_path)

    lm_trainer.config.save_config(log_dir / "trainer_config.json")
    lm_trainer.model.config.save_config(log_dir / "model_config.json")
    return train_losses


@torch.inference_mode()
def eval_model(
    lm_evaluator: Evaluator,
    test_loader: Union[data_utils.LogDataLoader, data_utils.TieredLogDataLoader],
    store_eval_data=False,
):
    """Perform testing on lm_trainer."""

    test_losses = []
    try:
        for iteration, batch in enumerate(tqdm(test_loader, desc="Test")):
            # Only allow interrupt between each batch
            with DelayedKeyboardInterrupt():
                split_batch = test_loader.split_batch(batch)
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
    except KeyboardInterrupt:
        # Proceed to evaluation
        print("Ctrl+C received, cancelling evaluation and exiting.")
        sys.exit(1)

    return test_losses
