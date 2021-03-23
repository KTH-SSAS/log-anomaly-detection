from argparse import ArgumentParser
import argparse

"""
Entrypoint script for training
"""

def train():
    pass

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("data-folder", type=str, help="Path to data files.")
    train(args)