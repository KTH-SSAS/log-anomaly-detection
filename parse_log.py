from argparse import ArgumentParser

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--char_lv', action='store_true',
                        help='Whether using character-level or word-level tokenization')
    parser.add_argument('-authfile',
                        type=str,
                        help='Path to an auth file.')
    parser.add_argument('-redfile',
                        type=str,
                        help='Path to a redteam file.')
    parser.add_argument('-outfile',
                        type=str,
                        help='Where to write derived features.')
    parser.add_argument('-outpath',
                        type=str,
                        help='Where to write output files.')
    args = parser.parse_args()
    return args

def tokenize_char(string, pad_len):
    """
    :param string:
    :param pad_len:
    :return:
    """
    return "0 " + " ".join([str(ord(c) - 30) for c in string]) + " 1 " + " ".join(["0"] * pad_len) + "\n"

def char_lv_tokenizer(args, weekend_days):
    LONGEST_LEN = 120  # Length of the longest line in auth_h.txt, used for padding

    with open(args.redfile, 'r') as red:
        redevents = set(red.readlines())

    with open(args.authfile, 'r') as infile:
        infile.readline()
        current_day = '0'
        day_outfile = open(args.outpath + current_day + '.txt', 'w')
        
        for line_num, line in enumerate(infile):
            if line_num % 10000 == 0:
                print(line_num)
            line_minus_time = ','.join(line.strip().split(',')[1:])
            pad_len = LONGEST_LEN - len(line_minus_time)
            raw_line = line.split(",")
            if len(raw_line) != 9: 
                print('bad length') 
                continue 
            sec = raw_line[0]
            user = raw_line[1].strip().split('@')[0]
            day = int(sec)//86400 # 24 hours * 60 minutes * 60 seconds 
            red = 0
            red += int(line in redevents)
            if user.startswith('U') and day not in weekend_days:
                index_rep = tokenize_char(line_minus_time, pad_len)
                current_line = f"{line_num} {sec} {day} {user.replace('U', '')} {red} {len(line_minus_time)+1} {index_rep}"
                larray = current_line.strip().split(' ')
                if int(larray[2]) == int(current_day):
                    day_outfile.write(current_line)
                else:
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
