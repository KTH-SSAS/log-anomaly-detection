import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch

import wandb
from log_analyzer.application import Application
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
        choices=["word-fields", "word-global", "word-merged", "char", "sentence"],
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


def main():
    # Initialize seeds
    set_seeds(22)
    args = prepare_args()

    if ("tiered" in args.model_type or "word" in args.tokenization) and args.counts_file is None:
        raise Exception("No field counts file was supplied!.")

    #  Start a W&B run

    os.environ["WANDB_MODE"] = "online" if args.wandb_sync else "offline"

    wandb.init(project="logml", entity="log-data-ml", config=vars(args))
    wandb_initalized = True

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

    # Create the trainer+model
    trainer, evaluator, train_loader, val_loader, test_loader = init_from_args(args)

    # If provided, load weights from file:
    if args.saved_model is not None:
        with open(args.saved_model, "rb") as f:
            logging.info("Loading weights from %s", args.saved_model)
            trainer.load_model_weights(f)

    # Train the model
    if not args.eval_only:
        train_model(trainer, train_loader, val_loader)
    # Test the model
    eval_model(evaluator, test_loader, store_eval_data=(not args.no_eval_model), model_file_name=args.saved_model)

    # Perform standard evaluation on the model
    if Application.instance().wandb_initialized and not args.no_eval_model:
        evaluator.run_all()


if __name__ == "__main__":
    main()
