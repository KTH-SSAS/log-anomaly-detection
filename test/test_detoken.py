import pytest
from log_analyzer.tokenizer.detokenizer import Int2Char, Int2Word

def dummy_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--type', 
                        choices=['char_level', 'word_level_count', 'word_level_translate', 'word_level_both'], 
                        required=False)
    parser.add_argument('-authfile',
                        type=str,
                        help='Path to an auth file.')
    parser.add_argument('-redfile',
                        type=str,
                        help='Path to a redteam file.')
    parser.add_argument('-outpath',
                        type=str,
                        help='Where to write output files.')
    parser.add_argument('-recordpath',
                        type=str,
                        help='Where to write record files.')
    parser.add_argument('-max_lines',
                        type=int,
                        default=None,
                        help='Maximum number of parsed lines.')
    args = parser.parse_args()
    return args


def test_Int2Char():
    
    args = dummy_arg_parser()
    example_log = '1,U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success'
    example_token = "0 55 19 18 19 34 38 49 47 19 14 37 19 26 24 20 6 34 38 49 47 19 14 37 19 26 24 20 14 37 19 26 24 20 14 33 14 33 14 35 87 86 74 47 67 82 14 53 87 69 69 71 85 85 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"	
    tokenizer = Char_tokenizer(args, None)
    detokenizer = Int2Char()
    
    line_minus_time = ','.join(example_log.strip().split(',')[1:])
    pad_len = 120 - len(line_minus_time)
    tokenized_log = tokenizer.tokenize_line(line_minus_time, pad_len)
    assert example_token == tokenized_log[:-1], "The example tokens and parsed log are not identical."
    detoken_example = detokenizer.run_detokenizer(ex_tokens)
    assert len(ex_tokens) != len(detoken_example), "the length of tokens does not match the length of outputs by character detokenization"
    

def test_Int2Word():
    example_log = '1,U101@DOM1,C1862$@DOM1,C1862,C1862,?,?,AuthMap,Success'
    example_token = "0 5 6 7 6 7 7 8 9 10 11 1"
    detoken_example = detokenizer.run_detokenizer(ex_tokens)
    assert len(ex_tokens) == len(detoken_example.split(',')), "the length of tokens does not match the length of of outputs by word detokenization"
