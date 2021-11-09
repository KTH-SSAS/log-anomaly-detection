import json
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class Config:
    """Parent class for configs."""

    def __init__(self) -> None:
        pass

    def save_config(self, filename):
        """save configuration as json file."""
        with open(filename, "w") as f:
            json.dump(self.__dict__, f, indent="\t")

    def load_config(self, filename):
        """load configuration from json file."""
        with open(filename, "r") as f:
            data = json.load(f)

        self.__dict__ = deepcopy(data)

    @classmethod
    def init_from_file(cls, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        return cls.init_from_dict(data)

    @classmethod
    def init_from_dict(cls, config_dict):
        return cls(**config_dict)
