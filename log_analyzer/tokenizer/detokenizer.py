import json

class Detokenizer:
    def __init__(self):
        self.skip_list = ['0', '1']

class Int2Char(Detokenizer):
    def __init__(self):
        super().__init__()

    def run_detokenizer(self, sentence):
        restored_lst = [chr(int(pred_token) + 30) for pred_token in sentence if pred_token not in self.skip_list]
        restored_txt = "".join(restored_lst)
        return restored_txt.split(',')

class Int2Word(Detokenizer):
    def __init__(self, dict_file):
        super().__init__()
