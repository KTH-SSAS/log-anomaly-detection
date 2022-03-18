from pathlib import Path

from log_analyzer.config import TrainerConfig


def test_save_load_config(tmpdir):

    config = TrainerConfig.init_from_file("config/lanl_config_trainer.json")
    savepath = Path(tmpdir) / "config.json"
    config.save_config(savepath)

    config_loaded = TrainerConfig.init_from_file(savepath)

    for (kv1, kv2) in zip(config.__dict__.items(), config_loaded.__dict__.items()):
        assert (kv1[0] == kv2[0]) and (kv1[1] == kv2[1])
