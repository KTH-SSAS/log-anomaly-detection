import json	

class Detokenizer:	
    def __init__(self):	
        self.skip_list = ['0', '1']	

class Int2Char(Detokenizer):	
    def __init__(self):	
        super().__init__()	

    def run_detokenizer(self, tokens):	
        restored_lst = [chr(t+30) for t in tokens]	
        restored_txt = "".join(restored_lst)	
        return restored_txt

class Int2Word(Detokenizer):	
    def __init__(self, dict_file):	
        super().__init__()	
        f = open(dict_file)	
        self.search_dict = json.load(f)	

    def run_detokenizer(self, tokens):	
        restored_lst = [self.search_dict[str(t)] for t in tokens]	
        restored_txt = ",".join(restored_lst)	
        return restored_txt	

if __name__ == "__main__":
    
    dict_file = 'data/data_examples/detoken_word/word_token_map.json'

    detokenizer = Int2Char()
    example = "0 55 19 18 19 34 38 49 47 19 14 37 19 26 24 20 6 34 38 49 47 19 14 37 19 26 24 20 14 37 19 26 24 20 14 33 14 33 14 35 87 86 74 47 67 82 14 53 87 69 69 71 85 85 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".split(' ')	
    ex_tokens = [int(t) for t in example]
    print(detokenizer.run_detokenizer(ex_tokens))

    detokenizer = Int2Word(dict_file)
    example = "0 5 6 7 6 7 7 8 9 10 11 1".split(' ')	
    ex_tokens = [int(t) for t in example]
    print(detokenizer.run_detokenizer(ex_tokens)) 