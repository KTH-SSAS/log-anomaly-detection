import pytest
import re
from argparse import ArgumentParser
from collections import Counter
from log_analyzer.tokenizer.detokenizer import Int2Char, Int2Word
from log_analyzer.tokenizer.tokenizer import Char_tokenizer

def test_Int2Char():
    log_line = '1,U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success'
    tokenized_log_line = "0 55 19 18 19 34 38 49 47 19 14 37 19 26 24 20 6 34 38 49 47 19 14 37 19 26 24 20 14 37 19 26 24 20 14 33 14 33 14 35 87 86 74 47 67 82 14 53 87 69 69 71 85 85 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"	
   
    detokenizer = Int2Char()
    
    line_minus_time = ','.join(log_line.strip().split(',')[1:])
    tokenized_log = detokenizer.run_tokenizer(line_minus_time, 120)

    assert tokenized_log_line == tokenized_log[:-1], "The example tokens and parsed log are not identical."
    
    assert detoken_example == line_minus_time, "The detokenized tokens and the original log input are not identical."

def test_Int2Word():
    log_line = '1,U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success'
    tokenized_log_line = "0 5 6 7 6 7 7 8 9 10 11 1"

    json_folder = 'data/data_examples/detoken_word/'
    detokenizer = Int2Word(json_folder)
    
    tokenized_log = detokenizer.run_tokenizer(log_line)
    assert tokenized_log_line == tokenized_log, "The example tokens and parsed log are not identical."

    ex_tokens = [int(t) for t in example_token.split(' ')]
    detoken_example = detokenizer.run_detokenizer(ex_tokens)
    detoken_example_minus_sos_eos = ','.join(detoken_example.split(',')[1:-1])
    ex_log_lst = re.split(r"[$@,]+",example_log)[1:]
    example_log_minus_time = ','.join(ex_log_lst)
    assert example_log_minus_time == detoken_example_minus_sos_eos, "The detokenized tokens and the original log input are not identical."
