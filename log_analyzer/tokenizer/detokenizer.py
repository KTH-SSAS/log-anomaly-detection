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