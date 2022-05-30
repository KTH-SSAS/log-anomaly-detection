"""Helper functions for model creation and training."""
import json
import logging
import os
import socket
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

    train_days = trainer_config.train_files
    test_days = trainer_config.test_files

    model_config.vocab_size = tokenizer.vocab_size
    model_config.sequence_length = calculate_max_input_length(task, tokenizer)

    if isinstance(model_config, TieredTransformerConfig):
        model_config.number_of_users = tokenizer.num_users

    if model_type in (TIERED_LSTM, TIERED_TRANSFORMER):
        val_loader = None
        train_loader, test_loader = data_utils.load_data_tiered(
            data_folder,
            train_days,
            test_days,
            (trainer_config.train_batch_size, trainer_config.eval_batch_size),
            tokenizer,
            task,
            num_steps=3,
        )
    elif model_type in (LSTM, TRANSFORMER):
        train_loader, val_loader, test_loader = data_utils.load_data(
            data_folder,
            train_days,
            test_days,
            (trainer_config.train_batch_size, trainer_config.eval_batch_size),
            tokenizer,
            task,
            trainer_config.validation_portion,
            shuffle_train_data,
        )
    elif model_type in (MULTILINE_TRANSFORMER) and isinstance(model_config, MultilineTransformerConfig):
        train_loader, val_loader, test_loader = data_utils.load_data_multiline(
            data_folder,
            train_days,
            test_days,
            (trainer_config.train_batch_size, trainer_config.eval_batch_size),
            tokenizer,
            task,
            model_config.shift_window,
            model_config.memory_type,
            trainer_config.validation_portion,
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

    @torch.inference_mode()
    def validation_run(train_iteration=0, val_run=0):
        """Performs one phase of validation on lm_trainer."""
        val_losses = []
        for val_iteration, val_batch in enumerate(tqdm(val_loader, desc=f"Valid:{val_run:2d}")):
            split_batch = val_loader.split_batch(val_batch)
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
        validation_period = (len(train_loader) // lm_trainer.config.validations_per_epoch) + 1
    else:
        validation_period = 0

    train_losses = []

    val_run = 0
    iteration = 0
    for epoch in tqdm(range(epochs), desc="Epoch   "):
        if (
            isinstance(train_loader.dataset, (data_utils.IterableLogDataset, data_utils.IterableUserMultilineDataset))
            and epoch > 0
        ):
            # Refresh the iterator so we can run another epoch
            train_loader.dataset.refresh_iterator()
        # Shuffle train data order for each epoch?
        # Count iteration continuously up through each epoch
        for epoch_iteration, batch in enumerate(tqdm(train_loader, desc="Training")):
            # epoch_iteration = iterations in this epoch (used to determine when to run validation)
            # Split the batch
            split_batch = train_loader.split_batch(batch)
            # Check that the split batch contains entries (see MultilineDataloader's mask filtering)
            if len(split_batch["X"]) == 0:
                continue
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
            if run_validation and epoch_iteration > 0 and (epoch_iteration % validation_period == 0):
                validation_run(iteration, val_run)
                val_run += 1

            if done:
                logger.info("Early stopping.")
                break

        if lm_trainer.epoch_scheduler is not None:
            lm_trainer.epoch_scheduler.step()

        if run_validation:
            validation_run(iteration, val_run)
            val_run += 1

        if done:
            break

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
    model_file_name=None,
):
    """Perform testing on lm_trainer.

    Note: model_file_name is only used for uploading model parameters to wandb.
    """

    if model_file_name is None:
        log_dir = lm_evaluator.checkpoint_dir
        model_file_name = "model.pt"
        model_save_path = log_dir / model_file_name
    else:
        model_save_path = model_file_name

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
        # Add any masked-out lines to the evaluator
        if "masked_batch" in split_batch:
            masked_batch = split_batch["masked_batch"]
            lm_evaluator.add_evaluation_data(
                masked_batch["user"],
                masked_batch["second"],
                masked_batch["red_flag"],
                mask=masked_batch["target_mask"],
            )
        # Check that the split batch contains entries (see MultilineDataloader's mask filtering)
        if len(split_batch["X"]) == 0:
            continue
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
