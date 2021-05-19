import json
from copy import deepcopy

class Config():
    """Parent class for confi"""
    def __init__(self) -> None:
        pass

    def save_config(self, filename):
        "save configuration as json file"
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)

    def load_config(self, filename):
        "load configuration from json file"
        with open(filename, 'r') as f:
            data = json.load(f)

        self.__dict__ = deepcopy(data)
