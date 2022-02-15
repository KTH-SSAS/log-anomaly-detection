import json
import os
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict

SECONDS_PER_DAY = 86400
import csv


class LANLReader:
    def __init__(self, file_pointer, already_split=False, has_red=False) -> None:
        self.field_names = [
            "time",
            "src_user",
            "src_domain",
            "dst_user",
            "dst_domain",
            "src_pc",
            "dst_pc",
            "auth_type",
            "logon_type",
            "auth_orient",
            "success",
        ]
        self._csv_field_names = [
            "time",
            "src_user@src_domain",
            "dst_user@dst_domain",
            "src_pc",
            "dst_pc",
            "auth_type",
            "logon_type",
            "auth_orient",
            "success",
        ]
        self.already_split = already_split
        self.has_red = has_red
        self.file_pointer = file_pointer

        if self.has_red:
            self._csv_field_names += "is_red"
            self.field_names += "is_red"

        if already_split:
            self._csv_field_names = self.field_names

    def __iter__(self):
        reader = csv.DictReader(self.file_pointer, fieldnames=self._csv_field_names)
        for row in reader:
            data = row

            if not self.already_split:
                for u, d in zip(["src_user", "dst_user"], ["src_domain", "dst_domain"]):
                    data[u], data[d] = data[f"{u}@{d}"].split("@")
                    del data[f"{u}@{d}"]

            data["dst_user"] = data["dst_user"].replace("$", "")

            yield data


def sec2day(seconds):
    day = int(seconds) // SECONDS_PER_DAY
    return day


def day2sec(day):
    return day * SECONDS_PER_DAY


def split_by_day(log_filename, out_dir, keep_days=None):
    """Split a raw LANL log file into separate days based on the timestamp.

    Also filters out non-user activity and splits the source/destination
    user/domain.
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
    """Split the [src|dst]_user and [src|dst]_domains into separate comma
    separated fields."""
    with open(outfile_path, "w") as outfile, open(infile_path, "r") as infile:
        reader = LANLReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.field_names)
        for entry in reader:
            writer.writerow(entry)


def add_redteam_to_log(filename_in, filename_out, readteam_file):
    """Adds redteam activity to a LANL log file as new field.

    The field is appended to each line.
    """

    with open(readteam_file) as f:
        redteam_events = f.readlines()

    redteam_events = [l for l in redteam_events if sec2day(l.split(",", maxsplit=1)[0]) == 8]

    with open(filename_out, "w") as outfile, open(filename_in, "r") as infile:
        reader = LANLReader(infile)
        writer = csv.DictWriter(outfile, reader.field_names + ["is_red"])
        for line in reader:

            red_style_line = ",".join((line["time"], line["src_user"], line["src_domain"], line["dst_domain"])) + "\n"

            if red_style_line in redteam_events:
                line["is_red"] = "1"
            else:
                line["is_red"] = "0"

            writer.writerow(line)


def process_logfiles_for_training(auth_file, red_file, output_dir, days_to_keep):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        split_by_day(auth_file, tmpdir, keep_days=days_to_keep)

        for day in days_to_keep:
            infile = os.path.join(tmpdir, f"{day}.csv")
            outfile = os.path.join(output_dir, f"{day}.csv")
            add_redteam_to_log(infile, outfile, red_file)


def count_fields(infile_path, outfile_path=None, fields_to_exclude=None):
    counts = OrderedDict()
    with open(infile_path) as f:
        reader = LANLReader(f)

        fields = reader.field_names
        for f in fields_to_exclude:
            del fields[f]

        for field in fields:
            counts[field] = {}

        for line in reader:
            for k in fields:
                v = line[k]
                try:
                    counts[k][v] += 1
                except KeyError:
                    counts[k][v] = 1

    if outfile_path is not None:
        with open(outfile_path, "w") as f:
            json.dump(counts, f)

    return counts


def process_file():
    parser = ArgumentParser()
    parser.add_argument("--auth-file", type=str, help="Path to auth.txt.")
    parser.add_argument("--redteam-file", type=str, help="Path to file with redteam events.")
    parser.add_argument("-o", "--output", type=str, help="Output directory.")
    parser.add_argument("days-to-keep", nargs="+", type=int, help="Days to keep logs from.")
    args = parser.parse_args()

    process_logfiles_for_training(args.auth_file, args.redteam_file, args.output, args.days_to_keep)
