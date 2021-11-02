import json
import os


class Detokenizer:
    def __init__(self):
        self.skip_list = ["0", "1"]


class Int2Char(Detokenizer):
    def __init__(self):
        super().__init__()

    def run_tokenizer(self, line_minus_time, total_len):
        """
        string: text log line (e.g., "U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success")
        output: tokenized log line (e.g., 0 55 19 18 19 34 38 49 47 19 ...... 74 47 67 82 14 53 87 69 69 71 85 85 1 0 0 0 0)
        """
        pad_len = total_len - len(line_minus_time)
        return "0 " + " ".join([str(ord(c) - 30) for c in line_minus_time]) + " 1 " + " ".join(["0"] * pad_len) + "\n"

    def run_detokenizer(self, tokens):
        """
        tokens: tokenized log lines (e.g., 0 55 19 18 19 34 38 49 47 19 ...... 74 47 67 82 14 53 87 69 69 71 85 85 1 0 0 0 0)
        restored_txt: text log line (e.g., U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success)
        """
        if isinstance(tokens, list):
            # skip 0 and 1 since they are <SOS> and <EOS>
            list_int_tokens = [int(t) for t in tokens if int(t) > 1]
        elif isinstance(tokens, str):
            # skip 0 and 1 since they are <SOS> and <EOS>
            list_int_tokens = [int(t) for t in tokens.split(" ") if int(t) > 1]
        restored_lst = [chr(t + 30) for t in list_int_tokens]
        restored_txt = "".join(restored_lst)
        return restored_txt


class Int2Word(Detokenizer):
    def __init__(self, json_folder):
        super().__init__()

        with open(os.path.join(json_folder, "word_token_map.json")) as json_file:
            self.search_dict = json.load(json_file)
        with open(os.path.join(json_folder, "usr_map.json")) as json_file:
            self.usr_inds = json.load(json_file)
        with open(os.path.join(json_folder, "pc_map.json")) as json_file:
            self.pc_inds = json.load(json_file)
        with open(os.path.join(json_folder, "domain_map.json")) as json_file:
            self.domain_inds = json.load(json_file)
        with open(os.path.join(json_folder, "auth_map.json")) as json_file:
            self.auth_dict = json.load(json_file)
        with open(os.path.join(json_folder, "logon_map.json")) as json_file:
            self.logon_dict = json.load(json_file)
        with open(os.path.join(json_folder, "orient_map.json")) as json_file:
            self.orient_dict = json.load(json_file)
        with open(os.path.join(json_folder, "success_map.json")) as json_file:
            self.success_dict = json.load(json_file)
        with open(os.path.join(json_folder, "other_map.json")) as json_file:
            self.other_inds = json.load(json_file)
        self.sos = 0
        self.eos = 1
        self.usr_OOV = 2
        self.pc_OOV = 3
        self.domain_OOV = 4
        self.curr_ind = 5

    def run_detokenizer(self, tokens):
        """
        tokens: tokenized log line (e.g., "0 5 6 7 6 7 7 8 9 10 11 1")
        restored_txt: text log line (e.g., "1,U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success")
        """
        if isinstance(tokens, list):
            list_int_tokens = [int(t) for t in tokens if int(t) > 1]
        elif isinstance(tokens, str):
            list_int_tokens = [int(t) for t in tokens.split(" ") if int(t) > 1]
        restored_lst = [self.search_dict[str(t)] for t in list_int_tokens]
        restored_txt = ",".join(restored_lst)
        return restored_txt

    def split_line(self, string):
        """
        string: text log line (e.g., 1,U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success)
        """
        data = string.strip().split(",")
        src_user = data[1].split("@")[0]
        src_domain = data[1].split("@")[1]
        dst_user = data[2].split("@")[0]
        dst_domain = data[2].split("@")[1]
        src_pc = data[3]
        dst_pc = data[4]
        return (
            src_user,
            src_domain,
            dst_user.replace("$", ""),
            dst_domain,
            src_pc,
            dst_pc,
        )

    def run_tokenizer(self, string):
        """
        string: text log line (e.g., "1,U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success")
        output: tokenized log line (e.g., "0 5 6 7 6 7 7 8 9 10 11 1")
        """

        data = string.split(",")

        src_user, src_domain, dst_user, dst_domain, src_pc, dst_pc = self.split_line(string)
        src_user = self.usr_inds[src_user]
        src_domain = self.domain_inds[src_domain]

        if dst_user.startswith("U"):
            dst_user = self.usr_inds[dst_user]
        else:
            dst_user = self.pc_inds[dst_user]
        dst_domain = self.domain_inds[dst_domain]

        src_pc = self.pc_inds[src_pc]
        dst_pc = self.pc_inds[dst_pc]

        # Deals with file corruption for this value.
        if data[5].startswith("MICROSOFT_AUTH"):
            data[5] = "MICROSOFT_AUTH"
        auth_type = self.auth_dict[data[5]]
        logon_type = self.logon_dict[data[6]]
        auth_orient = self.orient_dict[data[7]]
        success = self.success_dict[data[8].strip()]
        output = f"{self.sos} {src_user} {src_domain} {dst_user} {dst_domain} {src_pc} {dst_pc} {auth_type} {logon_type} {auth_orient} {success} {self.eos}"
        return output
