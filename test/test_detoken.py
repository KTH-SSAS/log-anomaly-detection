import pytest
from log_analyzer.tokenizer.detokenizer import Int2Char, Int2Word


def test_Int2Char():
    detokenizer = Int2Char()
    example = "0 55 19 18 19 34 38 49 47 19 14 37 19 26 24 20 6 34 38 49 47 19 14 37 19 26 24 20 14 37 19 26 24 20 14 33 14 33 14 35 87 86 74 47 67 82 14 53 87 69 69 71 85 85 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0".split(' ')	
    ex_tokens = [int(t) for t in example]
    detoken_example = detokenizer.run_detokenizer(ex_tokens)
    assert len(ex_tokens) != len(detoken_example), "the length of tokens does not match the length of outputs by character detokenization"
    

def test_Int2Word():
    dict_file = 'data/data_examples/detoken_word/word_token_map.json'
    detokenizer = Int2Word(dict_file)
    example = "0 5 6 7 6 7 7 8 9 10 11 1".split(' ')	
    ex_tokens = [int(t) for t in example]
    detoken_example = detokenizer.run_detokenizer(ex_tokens)
    assert len(ex_tokens) == len(detoken_example.split(',')), "the length of tokens does not match the length of of outputs by word detokenization"
