import json
from cmath import log

import numpy as np
import pytest
import torch

from log_analyzer.data.log_file_utils import count_fields
from log_analyzer.tokenizer.tokenizer_neo import LANLTokenizer, LANLVocab


def test_counter(log_file):

    counts = count_fields(log_file, fields_to_exclude=[0])
    assert counts["src_user"]["U22"] == 8
    assert counts["dst_user"]["U22"] == 8
    assert counts["src_domain"]["DOM1"] == 97


@pytest.mark.parametrize(
    "line",
    [
        ["U24", "DOM1", "U24", "DOM1", "C2198", "TGT", "?", "?", "TGS", "Success"],
        "U24,DOM1,U24,DOM1,C2198,TGT,?,?,TGS,Success",
        pytest.param("691200,U24@DOM1,U24@DOM1,C2198,TGT,?,?,TGS,Success", marks=pytest.mark.xfail),
    ],
)
def test_tokenizer(tokenizer: LANLTokenizer, line):

    indexes = tokenizer.tokenize(line)
    expected = np.array([6, 9, 12, 15, 18, 21, 24, 27, 30, 33])
    assert (indexes == expected).all()


def test_counts2vocab(counts_file):

    vocab = LANLVocab.counts2vocab(counts_file, "vocab.json", 0)

    assert vocab["special_tokens"]["[PAD]"] == 0
    assert "U24" in vocab["src_user"]


def test_dataloader(tokenizer):

    assert True


@pytest.mark.parametrize(
    "seed,num_masked_positions,expected_num_mask_tokens",
    [(6, 5, 5), (7, 5, 4), (8, 5, 3)],
)
def test_mask_tokens(tokenizer, seed, num_masked_positions, expected_num_mask_tokens):
    """This test runs the input masking function for different seeds, with
    different expected outputs.

    For seed=6, no mask tokens are changed to regular tokens or random
    tokens For seed=7, a single mask token is changed back to a regular
    token For seed=8, one mask token is changed back, and one is set to
    a random token
    """
    line = "U24,DOM1,U24,DOM1,C2198,TGT,?,?,TGS,Success"
    tokens = tokenizer.tokenize(line)

    np.random.seed(seed)

    masked_tokens, labels, sample_weights = tokenizer.mask_tokens(tokens, percentage_to_mask=0.5)

    assert sum(sample_weights) == num_masked_positions

    # Label is equal to original tokens
    assert all(labels == tokens)

    mask_indexes = tokenizer.vocab.mask_tokens
    num_mask_tokens = len([1 for i in masked_tokens if i in mask_indexes])
    assert num_mask_tokens == expected_num_mask_tokens
    assert True


def test_new_dataloader(tokenizer, processed_log_file):
    from torch.utils.data import DataLoader

    from log_analyzer.data.data_loader import collate_fn
    from log_analyzer.data.loader_neo import IterableLANLDataset

    np.random.seed(5)
    ds = IterableLANLDataset(processed_log_file, task=2, tokenizer=tokenizer)
    batch_size = 10
    num_fields = 10
    dl = DataLoader(ds, collate_fn=collate_fn, batch_size=batch_size)

    for batch in dl:
        assert batch["input"].shape == torch.Size((batch_size, num_fields))
        assert not torch.all(batch["red"])

        pass


def test_log_processing(auth_file):
    from log_analyzer.data.log_file_utils import process_logfiles_for_training

    process_logfiles_for_training(
        auth_file,
    )
