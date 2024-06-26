import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    """Parent class for configs."""

    def __init__(self, **_kwargs) -> None:
        pass

    def save_config(self, filename):
        """Save configuration as json file."""
        with open(filename, "w", encoding="utf8") as f:
            dictionary = self.__dict__

            for key, item in dictionary.items():
                if isinstance(item, Config):
                    dictionary[key] = asdict(item)

            json.dump(dictionary, f, indent="\t")

    def load_config(self, filename):
        """Load configuration from json file."""
        with open(filename, "r", encoding="utf8") as f:
            data = json.load(f)

        self.__dict__ = deepcopy(data)

    @classmethod
    def init_from_file(cls, filename: Path):
        with open(filename, "r", encoding="utf8") as f:
            data = json.load(f)

        return cls.init_from_dict(data)

    @classmethod
    def init_from_dict(cls, config_dict):
        if "_vocab_size" in config_dict:
            config_dict.pop("_vocab_size")
        if "sequence_length" in config_dict:
            config_dict.pop("sequence_length")
        return cls(**config_dict)
