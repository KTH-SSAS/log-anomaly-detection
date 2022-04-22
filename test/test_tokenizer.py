from pathlib import Path

import numpy as np
import pytest

from log_analyzer.data.log_file_utils import count_fields, process_logfiles_for_training
from log_analyzer.tokenizer.tokenizer_neo import CharTokenizer, GlobalTokenizer, LANLTokenizer, LANLVocab
from log_analyzer.tokenizer.vocab import GlobalVocab, MergedLANLVocab
from log_analyzer.train_loop import get_tokenizer


def test_counter(processed_log_file):

    counts = count_fields(processed_log_file, fields_to_exclude=[0], has_red=True)
    assert counts["src_user"]["U22"] == 8
    assert counts["dst_user"]["U22"] == 8
    assert counts["src_domain"]["DOM1"] == 97


def test_counts2vocab(counts_file):

    vocab = LANLVocab.counts2vocab(counts_file, 0)

    assert vocab.special_tokens["[PAD]"] == 0
    assert "U1" in vocab.vocab[0]


@pytest.mark.parametrize(
    "seed,num_masked_positions,expected_num_mask_tokens", [(6, 5, 5), (7, 5, 4), (8, 5, 3)],
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
    assert all(labels == sample_weights * tokens)

    mask_indexes = tokenizer.vocab.mask_tokens
    num_mask_tokens = len([1 for i in masked_tokens if i in mask_indexes])
    assert num_mask_tokens == expected_num_mask_tokens
    assert True


def test_log_processing(tmp_path, auth_file, redteam_file):
    """Test auth processing."""

    outfile = Path(tmp_path) / "out"
    outfile.mkdir()

    process_logfiles_for_training(auth_file, redteam_file, outfile, [0])


@pytest.mark.parametrize(
    "tokenization,expected",
    [
        (
            "word-global",
            (GlobalTokenizer, GlobalVocab, "U1053,DOM1,U1053,DOM1,[OOV],C625,Kerberos,Network,LogOn,Success"),
        ),
        ("word-fields", (LANLTokenizer, LANLVocab, "[OOV],DOM1,[OOV],DOM1,[OOV],C625,Kerberos,Network,LogOn,Success")),
        (
            "word-merged",
            (LANLTokenizer, MergedLANLVocab, "U1053,DOM1,U1053,DOM1,[OOV],C625,Kerberos,Network,LogOn,Success"),
        ),
        ("char", (CharTokenizer, None, "U1053,DOM1,U1053,DOM1,C862,C625,Kerberos,Network,LogOn,Success")),
    ],
)
@pytest.mark.parametrize("tiered", [True, False])
def test_tokenizers(tokenization, tiered, counts_file, expected):

    tokenizer = get_tokenizer(tokenization, tiered, counts_file, cutoff=2)

    e_tokenizer, e_vocab, expected_decoded = expected

    assert isinstance(tokenizer, e_tokenizer)
    if tokenization != "char":
        assert isinstance(tokenizer.vocab, e_vocab)

    data = "U1053,DOM1,U1053,DOM1,C862,C625,Kerberos,Network,LogOn,Success"
    indexes = tokenizer.tokenize(data)

    if tokenization == "char":
        assert len(indexes) == len(data)
    else:
        assert len(indexes) == len(data.split(","))

    data_decoded = tokenizer.detokenize(indexes)

    assert expected_decoded == data_decoded
