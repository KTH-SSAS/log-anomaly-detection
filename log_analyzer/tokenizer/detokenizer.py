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
    def __init__(self, json_folder, dict_file):	
        super().__init__()	

        with open(os.path.join(json_folder, dict_file)) as json_file:
            self.search_dict = json.load(json_file)
        with open(os.path.join(json_folder, 'usr_map.json')) as json_file:
            self.usr_inds = json.load(json_file)
        with open(os.path.join(json_folder, 'pc_map.json')) as json_file:
            self.pc_inds = json.load(json_file)
        with open(os.path.join(json_folder, 'domain_map.json')) as json_file:
            self.domain_inds = json.load(json_file)
        with open(os.path.join(json_folder, 'auth_map.json')) as json_file:
            self.auth_dict = json.load(json_file)
        with open(os.path.join(json_folder, 'logon_map.json')) as json_file:
            self.logon_dict = json.load(json_file)
        with open(os.path.join(json_folder, 'orient_map.json')) as json_file:
            self.orient_dict = json.load(json_file)
        with open(os.path.join(json_folder, 'success_map.json')) as json_file:
            self.success_dict = json.load(json_file)
        with open(os.path.join(json_folder, 'other_map.json')) as json_file:
            self.other_inds = json.load(json_file)
        self.sos = 0
        self.eos = 1
        self.usr_OOV = 2
        self.pc_OOV = 3
        self.domain_OOV = 4
        self.curr_ind = 5

    def run_detokenizer(self, tokens):	
        restored_lst = [self.search_dict[str(t)] for t in tokens]	
        restored_txt = ",".join(restored_lst)	
        return restored_txt	