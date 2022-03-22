import json
import operator
import os
import shutil
import sys
from pathlib import Path


def split_line(string):
    """Turn raw some fields of raw log line from auth_h.txt into a list of word
    tokens (needed for consistent user ids and domain ids)

    :param string: Raw log line from auth_h.txt
    :return: (list) word tokens for some fields of auth_h.txt
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


class CharTokenizer:
    def __init__(self, args, weekend_days):
        self.outpath: Path = Path(args.outpath)
        self.authfile = args.authfile
        self.redfile = args.redfile
        self.recordpath = args.recordpath
        self.max_lines = args.max_lines
        self.weekend_days = weekend_days
        self.LONGEST_LEN = 120  # Length of the longest line in auth.txt, used for padding
        self.current_day = None

    def build_output_dir(self):
        try:
            os.makedirs(self.outpath)
            print("Directory ", self.outpath, " Created ")
        except FileExistsError:
            print("Directory ", self.outpath, " already exists")

    @classmethod
    def tokenize_line(cls, string, pad_len):
        """
        :param string:
        :param pad_len:
        :return:
        """
        return "0 " + " ".join([str(ord(c) - 30) for c in string]) + " 1 " + " ".join(["0"] * pad_len) + "\n"

    @classmethod
    def detokenize_line(cls, tokens):
        return "".join([chr(t + 30) for t in tokens])

    def delete_duplicates(self):
        try:
            for filename in os.listdir(self.outpath):
                file_path = self.outpath / filename

                if file_path.is_file() or file_path.is_symlink():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)

        except FileNotFoundError:
            print("Nothing to delete.")

    def save_day_outfile(self, day_outfile, current_file_day, current_line, line_day):
        if int(line_day) == int(current_file_day):
            day_outfile.write(current_line)
        else:
            day_outfile.close()
            current_file_day = str(line_day)
            temp_path = self.outpath / (current_file_day + ".txt")
            if temp_path.is_file():
                # If the file exists, reopen the file and append new lines.
                day_outfile = open(temp_path, "a", encoding="utf8")
            else:
                # If the file doesn't exist, make new file..
                day_outfile = open(temp_path, "w", encoding="utf8")
            day_outfile.write(current_line)

    def tokenize(self):
        with open(self.redfile, "r", encoding="utf8") as red:
            redevents = set(red.readlines())

        with open(self.authfile, "r", encoding="utf8") as infile:
            infile.readline()  # Skip the first line.

            self.current_day = "0"
            day_outfile = open(self.outpath / (self.current_day + ".txt"), "w", encoding="utf8")

            for line_num, line in enumerate(infile):
                if line_num % 10000 == 0:
                    print(line_num)
                line_minus_time = ",".join(line.strip().split(",")[1:])
                pad_len = self.LONGEST_LEN - len(line_minus_time)
                raw_line = line.split(",")
                if len(raw_line) != 9:
                    print("bad length")
                    continue
                sec = raw_line[0]
                user = raw_line[1].strip().split("@")[0]
                day = int(sec) // 86400  # 24 hours * 60 minutes * 60 seconds
                red = 0
                # Reconstruct 'line' in the format of lines in redevents
                # Line format:
                # second,src_user@src_domain,dst_user@dst_domain,src_pc,dst_pc,auth_type,logon,auth_orient,success
                # redevent_line format:
                # second,src_user@src_domain,src_pc,dst_pc\n
                red_style_line = ",".join((sec, raw_line[1].strip(), raw_line[3], raw_line[4])) + "\n"
                red += int(red_style_line in redevents)
                if user.startswith("U") and day not in self.weekend_days:
                    index_rep = self.tokenize_line(line_minus_time, pad_len)
                    current_line = (
                        f"{line_num} {sec} {day} {user.replace('U', '')} {red} {len(line_minus_time)+1} {index_rep}"
                    )
                    self.save_day_outfile(day_outfile, self.current_day, current_line, day)
                if self.max_lines is not None:
                    if line_num > self.max_lines:
                        break
            day_outfile.close()

    def prepare_routes(self, _):
        self.delete_duplicates()
        self.build_output_dir()


class WordTokenizer(CharTokenizer):
    def __init__(self, args, weekend_days):
        super().__init__(args, weekend_days)
        self.OOV_CUTOFF = 40
        self.sos, self.eos, self.usr_OOV = 0, 1, 2
        self.pc_OOV = 3
        self.domain_OOV = 4
        self.curr_ind = 5
        # Placeholders to count occurences of all words.
        self.usr_counts = {}
        self.pc_counts = {}
        self.domain_counts = {}
        # Placeholders to save the pairs of each word and token.
        self.usr_inds = {}
        self.pc_inds = {}
        self.domain_inds = {}
        self.auth_dict = {}
        self.logon_dict = {}
        self.orient_dict = {}
        self.success_dict = {}
        self.other_inds = {
            "sos": self.sos,
            "eos": self.eos,
            "usr_OOV": self.usr_OOV,
            "pc_OOV": self.pc_OOV,
            "domain_OOV": self.domain_OOV,
        }
        self.path_usr_cnts = self.recordpath / "usr_counts"
        self.path_pc_cnts = self.recordpath / "pc_counts"
        self.path_domain_cnts = self.recordpath / "domain_counts"

    def build_record_dir(self):
        try:
            os.makedirs(self.recordpath)
            print("Directory ", self.recordpath, " Created ")
        except FileExistsError:
            print("Directory ", self.recordpath, " already exists")

    @classmethod
    def increment_freq(cls, ind_dict, key):
        """Used during -make_counts to track the frequencies of each element.

        :param ind_dict: (dict) keys: Raw word token, values: integer representation
        :param key: Raw word token
        """
        if key in ind_dict:
            ind_dict[key] += 1
        else:
            ind_dict[key] = 1

    def get_line_counts(self, line):

        data = line.strip().split(",")
        if len(data) != 9:
            return

        src_user, src_domain, dst_user, dst_domain, src_pc, dst_pc = split_line(line)

        self.increment_freq(self.usr_counts, src_user)
        self.increment_freq(self.domain_counts, src_domain)
        self.increment_freq(self.domain_counts, dst_domain)
        if dst_user.startswith("U"):
            self.increment_freq(self.usr_counts, dst_user)
        else:
            self.increment_freq(self.pc_counts, dst_user)
        self.increment_freq(self.pc_counts, dst_pc)
        self.increment_freq(self.pc_counts, src_pc)

    def count_words(self):

        with open(self.authfile, "r", encoding="utf8") as infile:
            infile.readline()
            for line_num, line in enumerate(infile):
                if line_num % 100000 == 0:
                    print(line_num)
                linevec = line.strip().split(",")
                user = linevec[1]
                day = int(linevec[0]) // 86400
                if user.startswith("U") and day not in self.weekend_days:
                    self.get_line_counts(line)
                if self.max_lines is not None:
                    if line_num > self.max_lines:
                        break

        self.write_sorted_counts(self.usr_counts, self.path_usr_cnts)
        self.write_sorted_counts(self.pc_counts, self.path_pc_cnts)
        self.write_sorted_counts(self.domain_counts, self.path_domain_cnts)

    @classmethod
    def write_sorted_counts(cls, count_dict, out_fn):
        """Sorts all of the elements in a dictionary by their counts and writes
        them to json and plain text.

        :param count_dict: (dict) keys: word tokens, values: number of occurrences
        :param out_fn: (str) Where to write .json and .txt files to (extensions are appended)
        """
        sorted_counts = sorted(count_dict.items(), key=operator.itemgetter(1))
        json_out_file = open(out_fn + ".json", "w", encoding="utf8")
        json.dump(count_dict, json_out_file)
        with open(out_fn + ".txt", "w", encoding="utf8") as outfile:
            for key, value in sorted_counts:
                outfile.write(f"{key}, {value}\n")

    def lookup(self, word, ind_dict, count_dict):
        """

        :param word: Raw text word token
        :param ind_dict: (dict) keys: raw word tokens, values: Integer representation
        :param count_dict: (dict) keys: raw word tokens, values: Number of occurrences
        :return: Integer representation of word
        """
        if count_dict is not None and count_dict[word] < self.OOV_CUTOFF:
            if count_dict is self.usr_counts:
                return self.usr_OOV
            if count_dict is self.pc_counts:
                return self.pc_OOV
            if count_dict is self.domain_counts:
                return self.domain_OOV

        if word not in ind_dict:
            ind_dict[word] = self.curr_ind
            self.curr_ind += 1
            return ind_dict[word]

        raise Exception("Word not found.")

    def translate_line(self, string, domain_counts, pc_counts):
        """Translates raw log line into sequence of integer representations for
        word tokens with sos and eos tokens.

        :param string: Raw log line from auth_h.txt
        :return: (list) Sequence of integer representations for word tokens with sos and eos tokens.
        """
        data = string.split(",")

        src_user, src_domain, dst_user, dst_domain, src_pc, dst_pc = split_line(string)
        src_user = self.lookup(src_user, self.usr_inds, None)
        src_domain = self.lookup(src_domain, self.domain_inds, domain_counts)

        if dst_user.startswith("U"):
            dst_user = self.lookup(dst_user, self.usr_inds, None)
        else:
            dst_user = self.lookup(dst_user, self.pc_inds, pc_counts)
        dst_domain = self.lookup(dst_domain, self.domain_inds, domain_counts)

        src_pc = self.lookup(src_pc, self.pc_inds, pc_counts)
        dst_pc = self.lookup(dst_pc, self.pc_inds, pc_counts)

        # Deals with file corruption for this value.
        if data[5].startswith("MICROSOFT_AUTH"):
            data[5] = "MICROSOFT_AUTH"
        auth_type = self.lookup(data[5], self.auth_dict, None)
        logon_type = self.lookup(data[6], self.logon_dict, None)
        auth_orient = self.lookup(data[7], self.orient_dict, None)
        success = self.lookup(data[8].strip(), self.success_dict, None)

        return (
            f"{self.sos} {src_user} {src_domain} {dst_user} {dst_domain} {src_pc} {dst_pc} {auth_type}"
            + f"{logon_type} {auth_orient} {success} {self.eos}\n"
        )

    def tokenize(self):

        try:
            if not self.usr_counts:
                with open(self.path_usr_cnts + ".json", encoding="utf8") as json_file:
                    self.usr_counts = json.load(json_file)
            if not self.pc_counts:
                with open(self.path_pc_cnts + ".json", encoding="utf8") as json_file:
                    self.pc_counts = json.load(json_file)
            if not self.domain_counts:
                with open(self.path_domain_cnts + ".json", encoding="utf8") as json_file:
                    self.domain_counts = json.load(json_file)
        except FileNotFoundError:
            print("No count files. Run word_level_count or word_level_both")
            sys.exit()

        current_day = "0"
        day_outfile = open(self.outpath / (current_day + ".txt"), "w", encoding="utf8")

        with open(self.redfile, "r", encoding="utf8") as red:
            redevents = set(red.readlines())

        with open(self.authfile, "r", encoding="utf8") as infile:
            for line_num, line in enumerate(infile):
                if line_num % 100000 == 0:
                    print(line_num)
                raw_line = line.split(",")
                if len(raw_line) != 9:
                    print("bad length")
                    continue
                sec = raw_line[0]
                user = raw_line[1].strip().split("@")[0]
                day = int(sec) // 86400
                red = 0
                # Reconstruct 'line' in the format of lines in redevents
                # Line format:
                # second,src_user@src_domain,dst_user@dst_domain,src_pc,dst_pc,auth_type,logon,auth_orient,success
                # redevent_line format:
                # second,src_user@src_domain,src_pc,dst_pc\n
                red_style_line = ",".join((sec, raw_line[1].strip(), raw_line[3], raw_line[4])) + "\n"
                red += int(red_style_line in redevents)
                if user.startswith("U") and day not in self.weekend_days:
                    index_rep = self.translate_line(line, self.domain_counts, self.pc_counts)
                    current_line = f"{line_num} {sec} {day} {user.replace('U', '')} {red} {index_rep}"
                    self.save_day_outfile(day_outfile, current_day, current_line, day)
                if self.max_lines is not None:
                    if line_num > self.max_lines:
                        break
            day_outfile.close()
        self.save_jsons()

    def save_jsons(self):

        with open(self.recordpath / (str(self.OOV_CUTOFF) + "_em_size.txt"), "w", encoding="utf8") as emsize_file:
            emsize_file.write(f"{self.curr_ind}")

        for data, file in zip(
            [
                self.usr_inds,
                self.pc_inds,
                self.domain_inds,
                self.auth_dict,
                self.logon_dict,
                self.orient_dict,
                self.success_dict,
                self.other_inds,
            ],
            [
                "usr_map.json",
                "pc_map.json",
                "domain_map.json",
                "auth_map.json",
                "logon_map.json",
                "orient_map.json",
                "success_map.json",
                "other_map.json",
            ],
        ):

            json.dump(data, open(self.recordpath / file, "w", encoding="utf8"))

        b_usr_inds = {v: k for k, v in self.usr_inds.items()}
        b_pc_inds = {v: k for k, v in self.pc_inds.items()}
        b_domain_inds = {v: k for k, v in self.domain_inds.items()}
        b_auth_inds = {v: k for k, v in self.auth_dict.items()}
        b_logon_inds = {v: k for k, v in self.logon_dict.items()}
        b_orient_inds = {v: k for k, v in self.orient_dict.items()}
        b_success_inds = {v: k for k, v in self.success_dict.items()}
        b_other_inds = {v: k for k, v in self.other_inds.items()}

        back_mappings = {
            **b_usr_inds,
            **b_pc_inds,
            **b_domain_inds,
            **b_auth_inds,
            **b_logon_inds,
            **b_orient_inds,
            **b_success_inds,
            **b_other_inds,
        }

        json.dump(
            back_mappings,
            open(self.recordpath / "word_token_map.json", "w", encoding="utf8"),
        )

    def prepare_routes(self, key):

        if key != "word_level_translate":
            self.delete_duplicates()
        self.build_output_dir()
        self.build_record_dir()
