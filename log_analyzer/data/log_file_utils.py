import os
from log_analyzer.tokenizer.log_analyzer import LANLReader

SECONDS_PER_DAY = 86400
import csv


def sec2day(seconds):
    day = int(seconds) // SECONDS_PER_DAY
    return day


def day2sec(day):
    return day * SECONDS_PER_DAY


def split_by_day(log_filename, out_dir, keep_days=None):
    """
    Split a raw LANL log file into separate days based on the timestamp.
    Also filters out non-user activity.
    """
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    current_day = 0
    get_filename = lambda day: os.path.join(out_dir, f"{day}.csv")
    out_file = None
    with open(log_filename) as f:
        reader = LANLReader(f)
        for line in reader:
            sec = line["time"]
            day = sec2day(sec)
            user = line["src_user"]

            if not (day in keep_days and user.startswith("U")):
                continue

            if day != current_day:
                current_day = day
                print(f"Processing day {current_day}...")
                try:
                    out_file.close()
                except AttributeError:
                    pass

                out_file = open(get_filename(current_day), "w")

            out_file.write(line)
    out_file.close()


def count_days():
    with open("data/tokenization_test_data/redteam.txt") as f:
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


def split_user_and_domain(infile_path, outfile_path):
    """
    Split the [src|dst]_user and [src|dst]_domains into separate comma separated fields.
    """
    with open(outfile_path, "w") as outfile, open(infile_path, "r") as infile:
        reader = LANLReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.field_names)
        for entry in reader:
            writer.writerow(entry)


def add_redteam_to_log(log_data_folder, day, readteam_file):
    """
    Adds redteam activity to a LANL log file as new field. The field is appended to each line.
    """
    with open(readteam_file) as f:
        redteam_events = f.readlines()

    redteam_events = [l for l in redteam_events if sec2day(l.split(",", maxsplit=1)[0]) == 8]

    with open(f"{day}_with_red.csv", "w") as outfile:
        with open(f"data/auth_by_day/{day}.csv") as f:
            reader = LANLReader(f)
            for line in reader:

                red_style_line = ",".join((line["time"], line["src_user"], line["src_domain"], line["dst_domain"])) + "\n"

                if red_style_line in redteam_events:
                    line["is_red"] = "1"
                else:
                    line["is_red"] = "0"

                outfile.write(",".join(split_line) + "\n")
