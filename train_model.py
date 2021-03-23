from argparse import ArgumentParser
import argparse

import log_analyzer.data.utils

"""
Entrypoint script for training
"""

def train():

    pass

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("data-folder", type=str, help="Path to data files.")
    train(args)