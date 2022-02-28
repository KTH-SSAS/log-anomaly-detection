import pytest

from log_analyzer.data.log_file_utils import add_redteam_to_log, count_fields
from log_analyzer.tokenizer.tokenizer_neo import LANLTokenizer, LANLVocab


@pytest.fixture()
def redteam_file(tmp_path):
    red_file = "data/test_data/redteam.txt"

    with open(red_file) as f:
        outfile = tmp_path / "redteam.txt"
        outfile.write_text(f.read())

    return outfile


@pytest.fixture()
def auth_file(tmp_path):
    filename = "data/test_data/auth_head.txt"

    with open(filename) as f:
        outfile = tmp_path / "auth.txt"
        outfile.write_text(f.read())

    return outfile


@pytest.fixture()
def raw_log_file(tmp_path):
    filename = "data/test_data/raw_8_head.csv"

    log_dir = tmp_path / "raw_logs"
    log_dir.mkdir()

    with open(filename, "r") as f:
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


@pytest.fixture()
def single_line_test_file(tmp_path):
    data = "691200,U24@DOM1,U24@DOM1,C2198,TGT,?,?,TGS,Success\n"
    log = tmp_path / "logfile.csv"
    log.write_text(data)
    return log


@pytest.fixture()
def counts_file(tmp_path, single_line_test_file):
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

    counts = count_fields(single_line_test_file, outfile_path=outfile, fields_to_exclude=[0], normalized=False)

    assert list(counts.keys()) == field_names[1:]

    assert counts["src_user"]["U24"] == 1
    assert counts["dst_domain"]["DOM1"] == 1

    return outfile


@pytest.fixture()
def vocab_file(tmp_path, counts_file):

    filename = tmp_path / "vocabfile.json"

    LANLVocab.counts2vocab(counts_file=counts_file, outfile=filename, cutoff=0)

    return filename


@pytest.fixture()
def tokenizer(vocab_file):
    vocab = LANLVocab(vocab_file)
    return LANLTokenizer(vocab)
