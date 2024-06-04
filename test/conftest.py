from pathlib import Path

import pytest
import torch

from log_analyzer.config.model_config import TieredTransformerConfig, TransformerConfig
from log_analyzer.data_utils.log_file_utils import add_redteam_to_log, count_fields
from log_analyzer.tokenizer.tokenizer_neo import LANLTokenizer, LANLVocab

SEQUENCE_LENGTH = 10
VOCAB_SIZE = 128
BATCH_SIZE = 4
CONSECUTIVE_LOG = 3
SHIFT_WINDOW = 3
LAYERS = 1
MODEL_DIM = 10
FFW_DIM = 10
DROPOUT_RATE = 0.1
ATTENTION_HEAD = 2
LEN_SAVED_HISTORY = 10
NUM_USERS = 100


@pytest.fixture()
def test_config():
    config = TransformerConfig(
        include_timestamp=False, layers=2, feedforward_dim=64, model_dim=64, attention_heads=2, dropout=0.1
    )
    config.vocab_size = VOCAB_SIZE
    config.sequence_length = SEQUENCE_LENGTH
    return config


@pytest.fixture
def test_tiered_transformer_config():
    args = {
        "layers": LAYERS,
        "feedforward_dim": FFW_DIM,
        "model_dim": MODEL_DIM,
        "attention_heads": ATTENTION_HEAD,
        "dropout": DROPOUT_RATE,
        "shift_window": SHIFT_WINDOW,
        "include_timestamp": False,
    }
    config = TieredTransformerConfig(**args)
    config.vocab_size = VOCAB_SIZE
    config.number_of_users = NUM_USERS
    config.sequence_length = SEQUENCE_LENGTH
    return config


@pytest.fixture()
def test_input():
    return torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQUENCE_LENGTH))


@pytest.fixture
def test_tiered_transformer_input():
    return (
        torch.randint(low=0, high=NUM_USERS, size=(BATCH_SIZE, 1)),
        torch.randint(low=0, high=VOCAB_SIZE, size=(CONSECUTIVE_LOG, BATCH_SIZE, SEQUENCE_LENGTH)),
    )


@pytest.fixture(name="redteam_file")
def fixture_redteam_file(tmp_path):
    red_file = "data/test_data/redteam.txt"

    with open(red_file, encoding="utf8") as f:
        outfile = tmp_path / "redteam.txt"
        outfile.write_text(f.read())

    return outfile


@pytest.fixture()
def auth_file(tmp_path):
    filename = "data/test_data/auth_head.txt"

    with open(filename, encoding="utf8") as f:
        outfile = tmp_path / "auth.txt"
        outfile.write_text(f.read())

    return outfile


@pytest.fixture(name="raw_log_file")
def fixture_raw_log_file(tmp_path):
    filename = "data/test_data/raw_8_head.csv"

    log_dir = tmp_path / "raw_logs"
    log_dir.mkdir()

    with open(filename, "r", encoding="utf8") as f:
        log_file = log_dir / "8.csv"
        log_file.write_text(f.read())

    return log_file


@pytest.fixture()
def processed_log_file(tmp_path, redteam_file, raw_log_file):

    out_dir = tmp_path / "red_logs"
    out_dir.mkdir()
    outfile = out_dir / "8.csv"

    add_redteam_to_log(8, raw_log_file, outfile, redteam_file)

    return outfile


@pytest.fixture(name="single_line_test_file")
def fixture_single_line_test_file(tmp_path):
    data = "691200,U24@DOM1,U24@DOM1,C2198,TGT,?,?,TGS,Success\n"
    log = tmp_path / "logfile.csv"
    log.write_text(data)
    return log


@pytest.fixture(name="counts_file")
def fixture_counts_file(tmp_path):
    field_names = [
        "time",
        "src_user",
        "src_domain",
        "dst_user",
        "dst_domain",
        "src_pc",
        "dst_pc",
        "auth_type",
        "logon_type",
        "auth_orient",
        "success",
    ]

    outfile = tmp_path / "countsfile.json"

    data_path = Path("data/test_data/")
    in_files = [data_path / "7.csv", data_path / "8.csv"]

    counts = count_fields(in_files, outfile_path=outfile, fields_to_exclude=[0, -1], normalized=True, has_red=True)

    assert list(counts.keys()) == field_names[1:]

    assert counts["src_user"]["U12"] == 3
    assert counts["src_user"]["U90"] == 13

    return outfile


@pytest.fixture(name="vocab")
def fixture_vocab(counts_file):

    vocab = LANLVocab.counts2vocab(counts=counts_file, cutoff=0)

    return vocab


@pytest.fixture()
def tokenizer(vocab):
    return LANLTokenizer(vocab)


@pytest.fixture
def context_history():
    return torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, LEN_SAVED_HISTORY, VOCAB_SIZE))
