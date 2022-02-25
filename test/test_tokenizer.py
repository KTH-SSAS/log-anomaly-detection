import numpy as np
import pytest

from log_analyzer.data.log_file_utils import count_fields, process_logfiles_for_training
from log_analyzer.tokenizer.tokenizer_neo import LANLTokenizer, LANLVocab


def test_counter(processed_log_file):

    counts = count_fields(processed_log_file, fields_to_exclude=[0], has_red=True)
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
    expected_detokenized = "U24,DOM1,U24,DOM1,C2198,TGT,?,?,TGS,Success"
    indexes = tokenizer.tokenize(line)
    expected = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
    assert (indexes == expected).all()

    detokenized_line = tokenizer.detokenize(indexes)
    assert detokenized_line == expected_detokenized


def test_counts2vocab(counts_file):

    vocab = LANLVocab.counts2vocab(counts_file, "vocab.json", 0)

    assert vocab.special_tokens["[PAD]"] == 0
    assert "U24" in vocab.vocab["src_user"]


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


def test_log_processing(tmp_path, auth_file, redteam_file):
    """Test auth processing."""

    outfile = tmp_path / "out"
    outfile.mkdir()

    process_logfiles_for_training(auth_file, redteam_file, outfile, [0])
