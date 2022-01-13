import os
from argparse import ArgumentParser

SECONDS_PER_DAY = 86400

def sec2day(seconds):
    day = int(seconds) // SECONDS_PER_DAY
    return day

def split_by_day(log_filename, out_dir, keep_days=None):

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    current_day = 0
    get_filename = lambda day: os.path.join(out_dir, f"{day}.csv")
    out_file = None
    with open(log_filename) as f:
        for line in f:
            fields = line.split(",")
            sec = fields[0]
            day = sec2day(sec)
            user = fields[1]

            if not user.startswith('U') or day not in keep_days:
                continue

            if day != current_day:
                print(current_day)
                current_day = day
                try:
                    out_file.close()
                except AttributeError:
                    pass
                out_file = open(get_filename(current_day), 'w')
            
            out_file.write(line)

def count_days():
    with open('data/tokenization_test_data/redteam.txt') as f:
        day_counts = {}
        for line in f:
            fields = line.split(",")
            sec = fields[0]
            day = sec2day(sec)

            try:
                day_counts[day] += 1
            except KeyError:
                day_counts[day] = 1

    print("Red team events by day:")
    print(day_counts)

def main():

    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("days_to_include", nargs='+')

    args = parser.parse_args()

    split_by_day(args.input, args.output_dir, keep_days=args.days_to_include)


if __name__ == '__main__':
    main()