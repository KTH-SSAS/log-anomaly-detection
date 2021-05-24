from argparse import ArgumentParser
import os, shutil
import json
import operator

class Char_tokenizer:
    def __init__(self, args, weekend_days):
        self.outpath = args.outpath
        self.authfile = args.authfile
        self.redfile = args.redfile
        self.record_dir = args.recordpath
        self.weekend_days = weekend_days
        self.LONGEST_LEN = 120 # Length of the longest line in auth.txt, used for padding

    def build_output_dir(self):
        try:
            os.makedirs(self.outpath)    
            print("Directory " , self.outpath ,  " Created ")
        except FileExistsError:
            print("Directory " , self.outpath ,  " already exists")  
        try:
            os.makedirs(self.record_dir)    
            print("Directory " , self.record_dir ,  " Created ")
        except FileExistsError:
            print("Directory " , self.record_dir ,  " already exists")  

    def tokenize_line(self, string, pad_len):
        """
        :param string:
        :param pad_len:
        :return:
        """
        return "0 " + " ".join([str(ord(c) - 30) for c in string]) + " 1 " + " ".join(["0"] * pad_len) + "\n"
        
    def delete_duplicates(self):
        for filename in os.listdir(self.outpath):
            file_path = os.path.join(self.outpath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def save_day_outfile(self, day_outfile, current_file_day, current_line, line_day):
        if int(line_day) == int(current_file_day):
            day_outfile.write(current_line)
        else:
            day_outfile.close()
            current_file_day = line_day
            if os.path.isfile(self.outpath + current_file_day + '.txt'):
                day_outfile = open(self.outpath + current_file_day + '.txt', 'a') # If the file exists, reopen the file and append new lines.
            else:
                day_outfile = open(self.outpath + current_file_day + '.txt', 'w') # If the file doesn't exist, make new file..
            day_outfile.write(current_line)

    def tokenize(self):
        with open(self.redfile, 'r') as red:
            redevents = set(red.readlines())

        with open(self.authfile, 'r') as infile:
            infile.readline() # Skip the first line.

            self.current_day = '0'
            day_outfile = open(self.outpath + self.current_day + '.txt', 'w')
            
            for line_num, line in enumerate(infile):
                if line_num % 10000 == 0:
                    print(line_num)
                line_minus_time = ','.join(line.strip().split(',')[1:])
                pad_len = self.LONGEST_LEN - len(line_minus_time)
                raw_line = line.split(",")
                if len(raw_line) != 9: 
            if len(raw_line) != 9: 
                if len(raw_line) != 9: 
                    print('bad length') 
                print('bad length') 
                    print('bad length') 
                    continue 
                continue 
                    continue 
                sec = raw_line[0]
                user = raw_line[1].strip().split('@')[0]
                day = int(sec) // 86400 # 24 hours * 60 minutes * 60 seconds 
                red = 0
                red += int(line in redevents)
                if user.startswith('U') and day not in weekend_days:
                    index_rep = self.tokenize_line(line_minus_time, pad_len)
                    current_line = f"{line_num} {sec} {day} {user.replace('U', '')} {red} {len(line_minus_time)+1} {index_rep}"
                    self.save_day_outfile(day_outfile, self.current_day, current_line, day)
            day_outfile.close()

    def run_tokenizer(self):

        self.delete_duplicates()
        self.build_output_dir()
        self.tokenize()

                    day_outfile.close()
                    current_day = larray[2]
                    day_outfile = open(args.outpath + current_day + '.txt', 'w')
                    day_outfile.write(current_line)
        day_outfile.close()

def word_lv_tokenizer(args, weekend_days):
    # TODO

if __name__ == '__main__':
    
    args = arg_parser()
    weekend_days = [3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46, 47, 52, 53]
    if args.char_lv:
        char_lv_tokenizer(args, weekend_days)
    else:
        word_lv_tokenizer(args, weekend_days)
