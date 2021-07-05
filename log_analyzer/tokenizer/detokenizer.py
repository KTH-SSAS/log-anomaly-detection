import json

class Detokenizer:
    def __init__(self):
        self.skip_list = ['0', '1']

class Int2Char(Detokenizer):
    def __init__(self):
        super().__init__()

class Int2Word(Detokenizer):
    def __init__(self, dict_file):
        super().__init__()
