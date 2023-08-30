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
    assert "U1" in vocab.vocab[1]


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
            (GlobalTokenizer, GlobalVocab, "T8,U1053,DOM1,U1053,DOM1,[OOV],C625,Kerberos,Network,LogOn,Success"),
        ),
        (
            "word-fields",
            (LANLTokenizer, LANLVocab, "T8,[OOV],DOM1,[OOV],DOM1,[OOV],C625,Kerberos,Network,LogOn,Success")
        ),
        (
            "word-merged",
            (LANLTokenizer, MergedLANLVocab, "T8,U1053,DOM1,U1053,DOM1,[OOV],C625,Kerberos,Network,LogOn,Success"),
        ),
        (
            "char",
            (CharTokenizer, None, "T8,U1053,DOM1,U1053,DOM1,C862,C625,Kerberos,Network,LogOn,Success")
        ),
    ],
)
@pytest.mark.parametrize("include_timestamp", [True, False])
def test_tokenizers(tokenization, counts_file, expected, include_timestamp):
    """Test tokenizers."""
    # Including timestamps is not supported for character tokenization
    if tokenization == "char" and include_timestamp:
        pytest.skip()

    tokenizer = get_tokenizer(tokenization, counts_file, cutoff=2, include_timestamp=include_timestamp)

    e_tokenizer, e_vocab, expected_decoded = expected
    if not include_timestamp:
        expected_decoded = ",".join(expected_decoded.split(",")[1:])

    assert isinstance(tokenizer, e_tokenizer)
    if tokenization != "char":
        assert isinstance(tokenizer.vocab, e_vocab)

    data = "U1053,DOM1,U1053,DOM1,C862,C625,Kerberos,Network,LogOn,Success"
    if include_timestamp:
        data = "635015," + data
    indexes = tokenizer.tokenize(data)

    if tokenization == "char":
        assert len(indexes) == len(data)
    else:
        assert len(indexes) == len(data.split(","))

    data_decoded = tokenizer.detokenize(indexes)

    assert expected_decoded == data_decoded
