import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

import wandb
from log_analyzer.application import Application
from log_analyzer.data.data_loader import MultilineDataLoader
from log_analyzer.train_loop import eval_model, init_from_args, train_model


def prepare_args():
    parser = ArgumentParser()
    parser.add_argument(
        "model_type", choices=["lstm", "tiered-lstm", "transformer", "tiered-transformer", "multiline-transformer"]
    )
    parser.add_argument(
        "tokenization",
        type=str,
        help="Tokenization method",
        choices=["word-fields", "word-global", "word-merged", "char"],
    )
    parser.add_argument("-mc", "--model-config", type=str, help="Model configuration file.", required=True)
    parser.add_argument(
        "-cf",
        "--counts-file",
        type=str,
        help="Path to field counts file. Required for field tokenization and tiered models.",
        required=False,
    )
    parser.add_argument("-df", "--data-folder", type=str, help="Path to data files.", required=True)
    parser.add_argument("-tc", "--trainer-config", type=str, help="Trainer configuration file.", required=True)
    parser.add_argument("--load-model", type=str, help="Path to saved model for initialization.", dest="saved_model")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluator.")
    parser.add_argument("--wandb-group", type=str, help="WANDB group to store run in.")
    parser.add_argument(
        "--bidir",
        dest="bidirectional",
        action="store_true",
        help="Use model in bidirectional mode. Only applies to lower level models when using tiered architectures.",
    )
    parser.add_argument("--model-dir", type=str, help="Directory to save stats and checkpoints to", default="runs")
    parser.add_argument(
        "--no-eval-model",
        action="store_true",
        help="Including this option will skip running the model through standard"
        "evaluation and returning appropriate metrics and plots.",
    )
    parser.add_argument(
        "--wandb-sync", action="store_true", help="Including this option will sync the wandb data with the cloud."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA acceleration for training.")
    args = parser.parse_args()
    return args


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def main(seed=22):
    # Initialize seeds
    set_seeds(seed)
    args = prepare_args()

    if ("tiered" in args.model_type or "word" in args.tokenization) and args.counts_file is None:
        raise Exception("No field counts file was supplied!.")

    #  Start a W&B run

    os.environ["WANDB_MODE"] = "online" if args.wandb_sync else "offline"

    eval_only: bool = args.eval_only
    saved_model: str = args.saved_model
    wandb_group = args.wandb_group
    evaluate: bool = not args.no_eval_model

    wandb.init(project="logml", entity="log-data-ml", config=vars(args), group=wandb_group)
    wandb_initalized = True

    # Log the seed
    wandb.config.update({"seed": seed})

    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA not available. Ignoring the --cuda option.")
        using_cuda = False
    else:
        using_cuda = args.use_cuda

    Application(cuda=using_cuda, wandb=wandb_initalized)

    if args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    logging.basicConfig(level=log_level)

    # Look for a trainer_config.json and model_config.json in the folder of the saved model. Use these if available
    if args.saved_model:
        folder_path = Path(args.saved_model)
        model_conf: Path = folder_path.parent / "model_config.json"
        trainer_conf: Path = folder_path.parent / "trainer_config.json"
        if model_conf.exists() and trainer_conf.exists():
            args.model_config = model_conf
            args.trainer_config = trainer_conf

    # Create the trainer+model
    trainer, evaluator, train_loader, val_loader, test_loader = init_from_args(args)

    def path_to_model(default_model_filename: str) -> Path:
        log_dir = evaluator.checkpoint_dir
        # If we only eval and have specified a model, we evaluate it
        if eval_only and saved_model:
            return Path(saved_model)
        # Otherwise we use the model that was trained
        return log_dir / default_model_filename

    model_to_evaluate = path_to_model("model_best.pt")

    def load_weights(path):
        with open(path, "rb") as f:
            logging.info("Loading weights from %s", path)
            trainer.load_model_weights(f)

    # Train the model
    if not eval_only:
        if saved_model:
            # Initialize training from saved weights if provided
            load_weights(saved_model)

        if isinstance(train_loader, MultilineDataLoader):
            # Set the training flag to handle incomplete sequences correctly
            train_loader.dataset.training = True

        train_model(trainer, train_loader, val_loader)

        if Application.instance().wandb_initialized and model_to_evaluate.exists():
            # Save the model weights as a versioned artifact
            artifact = wandb.Artifact(
                Application.artifact_name,
                "model",
                metadata=evaluator.model.config.__dict__,
            )
            artifact.add_file(model_to_evaluate)
            artifact.save()

    if model_to_evaluate.exists():
        # Load the weights of the model to evaluate
        load_weights(model_to_evaluate)
        # Otherwise the version of the model updated last will be evaluated

    # Remove unused memory
    del trainer, train_loader, val_loader

    # Test the model
    eval_model(evaluator, test_loader, store_eval_data=evaluate)

    # Perform standard evaluation on the model
    if Application.instance().wandb_initialized and evaluate:
        evaluator.run_all()

    wandb.finish()
    Application.reset()


if __name__ == "__main__":

    for seed_val in range(1, 2):
        main(seed_val)
