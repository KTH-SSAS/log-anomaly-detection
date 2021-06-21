from argparse import ArgumentParser
from log_analyzer.tokenizer.tokenizer import Char_tokenizer, Word_tokenizer
      
def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--type', 
                        choices=['char_level', 'word_level_count', 'word_level_translate', 'word_level_both'], 
                        required=True)
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

if __name__ == '__main__':
    
    args = arg_parser()
    weekend_days = [3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46, 47, 52, 53]
    if args.type == 'char_level':
        tokenizer = Char_tokenizer(args, weekend_days)
        tokenizer.prepare_routes()
    else:
        tokenizer = Word_tokenizer(args, weekend_days)
        tokenizer.prepare_routes(args.type)

    if args.type == 'word_level_count':
        tokenizer.count_words()
    elif args.type in ['char_level', 'word_level_translate']: 
        tokenizer.tokenize()
    elif args.type == 'word_level_both':
        tokenizer.count_words()
        tokenizer.tokenize()

"""
In order to run it for the initial *n* lines of the file, you need to specify the number of lines you want to process after -max_lines.
If you want to run the below code for an entire file, you need to remove the line starts with -max_lines.

1. For char-level tokenization: 

    python parse_log.py
    --type char_level
    -authfile data/auth.txt
    -redfile data/redteam.txt
    -outpath parsed_data/char_token/
    -recordpath parsed_data/char_token/records/
    -max_lines 1000 
                    
2. For word-level tokenization, there are three settings:

    a. Only counting the occurences of words and generating json files of occurences:

        python parse_log.py
        --type word_level_count
        -authfile data/auth.txt
        -redfile data/redteam.txt
        -outpath parsed_data/word_token/
        -recordpath parsed_data/word_token/records/
        -max_lines 1000

    b. Reading the json files of occurences and running translation:

        python parse_log.py
        --type word_level_translate
        -authfile data/auth.txt
        -redfile data/redteam.txt
        -outpath parsed_data/word_token/
        -recordpath parsed_data/word_token/records/
        -max_lines 1000

    c. Run a whole code from beginning to end:

        python parse_log.py
        --type word_level_both
        -authfile data/auth.txt
        -redfile data/redteam.txt
        -outpath parsed_data/word_token/
        -recordpath parsed_data/word_token/records/
        -max_lines 1000 
"""