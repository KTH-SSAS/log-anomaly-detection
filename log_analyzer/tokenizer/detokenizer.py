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
        f = open(dict_file)
        self.search_dict = json.load(f)
        self.search_dict[0] = "<SOS>"
        self.search_dict[1] = "<EOS>"
        self.search_dict[2] = "<usr_OOV>"
        self.search_dict[3] = "<pc_OOV>"
        self.search_dict[4] = "<domain_OOV>"

    def run_detokenizer(self, sentence):
        restored_txt = [self.search_dict[pred_token] for pred_token in sentence if pred_token not in self.skip_list]
        return restored_txt

if __name__ == "__main__":

    dict_file = 'parsed_data/word_token/records/word_token_map.json'

    detokenizer = Int2Char()
    ex_prediction = "0 55 19 18 19 34 38 49 47 19 14 37 19 26 24 20 6 34 38 49 47 19 14 37 19 26 24 20 14 37 19 26 24 20 14 33 14 33 14 35 87 86 74 47 67 82 14 53 87 69 69 71 85 85 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".split(' ')
    print(detokenizer.run_detokenizer(ex_prediction))

    detokenizer = Int2Word(dict_file)
    ex_prediction = "0 5 6 7 6 7 7 8 9 10 11 1".split(' ')
    print(detokenizer.run_detokenizer(ex_prediction))