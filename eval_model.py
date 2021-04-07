from argparse import ArgumentParser

def eval():
    #TODO evaluation code
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('model_checkpoint', type=str, help='Path path to model checkpoint that should be evaluated')
    parser.add_argument("--data-folder", type=str, help="Path to data files.")
    parser.add_argument('--jagged', action='store_true',
                        help='Whether using sequences of variable length (Input should'
                             'be zero-padded to max_sequence_length.')
    parser.add_argument('--skipsos', action='store_true',
                        help='Whether to skip a start of sentence token.')
    parser.add_argument('--bidir', dest='bidirectional', action='store_true',
                        help='Whether to use bidirectional lstm for lower tier.')
    parser.add_argument('-bs', dest='batch_size', type=int, help='batch size.')
    parser.add_argument('--config', type=str, default='config.json', help='JSON configuration file')
    args = parser.parse_args()
    eval(args)