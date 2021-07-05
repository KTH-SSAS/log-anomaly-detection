import json

class Detokenizer:
    def __init__(self):
        self.skip_list = ['0', '1']

class Int2Char(Detokenizer):
    def __init__(self):
        super().__init__()

