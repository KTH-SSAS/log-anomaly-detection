import csv
import json
import os
import sys
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict

from log_analyzer.tokenizer.tokenizer_neo import LANLVocab
from log_analyzer.tokenizer.vocab import GlobalVocab

SECONDS_PER_DAY = 86400


class LANLReader:
    """Reader class for parsing LANL log data."""

    def __init__(self, file_pointer, normalized=True, has_red=False) -> None:
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
        self.normalized = normalized
        self.has_red = has_red
        self.file_pointer = file_pointer

        if self.has_red:
            self._csv_field_names.append("is_red")
            self.field_names.append("is_red")

        if normalized:
            self._csv_field_names = self.field_names

    def __iter__(self):
        reader = csv.DictReader(self.file_pointer, fieldnames=self._csv_field_names)
        for row in reader:

            if row[self.field_names[-1]] is None or None in row:
                raise RuntimeError("The number of fields in the data does not match the settings provided.")

            data = row

            if not self.normalized:
                for u, d in zip(["src_user", "dst_user"], ["src_domain", "dst_domain"]):
                    data[u], data[d] = data[f"{u}@{d}"].split("@")
                    del data[f"{u}@{d}"]

            data["dst_user"] = data["dst_user"].replace("$", "")

            yield data


def sec2day(seconds):
    """Seconds to number of whole days."""
    day = int(seconds) // SECONDS_PER_DAY
    return day


def day2sec(day):
    """Day to number of seconds."""
    return day * SECONDS_PER_DAY


def split_by_day(log_filename, out_dir, keep_days=None):
    """Split a raw LANL log file into separate days based on the timestamp.

    Also filters out non-user activity and splits the source/destination
    user/domain.
    """
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    current_day = -1

    def get_filename(day):
        return os.path.join(out_dir, f"{day}.csv")

    out_file = None
    with open(log_filename, encoding="utf8") as f:
        for line in f:
            split_line = line.strip().split(",", maxsplit=2)
            sec = split_line[0]
            day = sec2day(sec)
            user = split_line[1]

            if not (day in keep_days and user.startswith("U")):
                continue

            if day > current_day:
                current_day = day
                print(f"Processing day {current_day}...")
                try:
                    out_file.close()
                except AttributeError:
                    pass

                out_file = open(get_filename(current_day), "w", encoding="utf8")
            elif day < current_day:
                raise RuntimeError
            else:
                pass

            out_file.write(line)
    if not out_file is None:
        out_file.close()


def count_events_per_day(redfile):
    """Count the number of red team events in the redteam file by day."""
    with open(redfile, encoding="utf8") as f:
        day_counts = {}
        for line in f:
            fields = line.split(",")
            sec = fields[0]
            day = sec2day(sec)

            # print(f"{day} : {line}")
            try:
                day_counts[day] += 1
            except KeyError:
                day_counts[day] = 1

    print("Red team events by day:")
    print(day_counts)


def split_user_and_domain(infile_path, outfile_path):
    """Split the [src|dst]_user and [src|dst]_domains into separate comma
    separated fields."""
    with open(outfile_path, "w", encoding="utf8") as outfile, open(infile_path, "r", encoding="utf8") as infile:
        reader = LANLReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.field_names)
        for entry in reader:
            writer.writerow(entry)


def add_redteam_to_log(day, filename_in, filename_out, readteam_file, normalized=False):
    """Adds redteam activity to a LANL log file as new field.

    The field is appended to each line.
    """

    with open(readteam_file, encoding="utf8") as f:
        redteam_events = f.readlines()

    redteam_events = [l for l in redteam_events if sec2day(l.split(",", maxsplit=1)[0]) == day]

    with open(filename_out, "w", encoding="utf8") as outfile, open(filename_in, "r", encoding="utf8") as infile:
        reader = LANLReader(infile, normalized=normalized)
        writer = csv.DictWriter(outfile, reader.field_names + ["is_red"])
        for line in reader:

            red_style_line = (
                f"""{line['time']},{line['src_user']}@{line['src_domain']},{line['src_pc']},{line['dst_pc']}\n"""
            )

            if red_style_line in redteam_events:
                line["is_red"] = "1"
            else:
                line["is_red"] = "0"

            writer.writerow(line)


def process_logfiles_for_training(auth_file, red_file, output_dir, days_to_keep):
    """Process auth.txt into normalized log files split into days."""
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        split_by_day(auth_file, tmpdir, keep_days=days_to_keep)

        for day in days_to_keep:
            infile = os.path.join(tmpdir, f"{day}.csv")
            outfile = os.path.join(output_dir, f"{day}.csv")
            add_redteam_to_log(day, infile, outfile, red_file)


def get_all_users(infile_path, outfile_path, has_red=True):
    users = set()

    if not isinstance(infile_path, list):
        infile_path = [infile_path]

    for file in infile_path:
        with open(file, encoding="utf8") as f:
            reader = LANLReader(f, normalized=True, has_red=has_red)
            for line in reader:
                users.add(line["src_user"])

    users = list(users)

    if outfile_path is not None:
        with open(outfile_path, "w", encoding="utf8") as f:
            json.dump(users, f)

    return users


def count_fields(infile_path, outfile_path=None, fields_to_exclude=None, normalized=True, has_red=False):
    """Count fields in the given file."""
    counts = OrderedDict()

    if not isinstance(infile_path, list):
        infile_path = [infile_path]

    for file in infile_path:
        print(f"Counting fields in {file}...")
        with open(file, encoding="utf8") as f:
            reader = LANLReader(f, normalized=normalized, has_red=has_red)

            fields = reader.field_names.copy()
            for f in fields_to_exclude:
                del fields[f]

            for line in reader:
                for field in fields:

                    value = line[field]

                    # Count PCs that appear in the "dst_user" as "dst_pc"
                    if field == "dst_user" and value.startswith("C"):
                        field = "dst_pc"

                    try:
                        field_counts = counts[field]
                    except KeyError:
                        counts[field] = {}
                        field_counts = counts[field]

                    try:
                        field_counts[value] += 1
                    except KeyError:
                        field_counts[value] = 1

        if outfile_path is not None:
            with open(outfile_path, "w", encoding="utf8") as f:
                json.dump(counts, f)

    return counts


def process_file():
    """CLI tool to process auth.txt."""
    parser = ArgumentParser()
    parser.add_argument("auth_file", type=str, help="Path to auth.txt.")
    parser.add_argument("redteam_file", type=str, help="Path to file with redteam events.")
    parser.add_argument("days_to_keep", nargs="+", type=int, help="Days to keep logs from.")
    parser.add_argument("-o", "--output", type=str, help="Output directory.", default="auth_split")
    args = parser.parse_args()
    process_logfiles_for_training(args.auth_file, args.redteam_file, args.output, args.days_to_keep)


def generate_counts(arguments=None):
    """CLI tool to generate counts file."""
    parser = ArgumentParser()
    parser.add_argument("log_files", type=str, nargs="+", help="Glob pattern of log files to process.")
    parser.add_argument(
        "--not-normalized", action="store_false", help="Add this flag if the log file is not already normalized."
    )
    parser.add_argument(
        "--no-red", action="store_false", help="Add this flag if the log file does not have red team events added."
    )
    parser.add_argument("-o", "--output", help="Output filename.", default="counts.json")
    parser.add_argument(
        "--fields-to-exclude", nargs="+", type=int, help="Indexes of fields to not count.", default=[0, -1]
    )

    if arguments is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arguments)

    # args = parser.parse_args(["data/train_data/7.csv", "--fields-to-exclude", "0"])
    count_fields(args.log_files, args.output, args.fields_to_exclude, args.not_normalized, args.no_red)


def count_users():
    """CLI tool to generate  file."""
    parser = ArgumentParser()
    parser.add_argument("log_files", type=str, nargs="+", help="Glob pattern of log files to process.")
    parser.add_argument(
        "--no-red", action="store_false", help="Add this flag if the log file does not have red team events added."
    )
    parser.add_argument("-o", "--output", help="Output filename.", default="users.json")

    args = parser.parse_args()

    get_all_users(args.log_files, args.output, args.no_red)


def generate_vocab_from_counts():
    """Generate vocab."""
    parser = ArgumentParser()
    parser.add_argument("counts_file", type=str, help="Path to JSON file with field counts.")
    parser.add_argument("mode", choices=["fields", "global"])
    parser.add_argument(
        "cutoff", type=int, help="If a token occurs less than the cutoff value, it will not be included."
    )
    parser.add_argument("-o", "--output", type=str, help="Output filename", default="vocab.json")

    args = parser.parse_args()

    if args.mode == "fields":
        vocab = LANLVocab.counts2vocab(args.counts_file, args.cutoff)
    elif args.mode == "global":
        vocab = GlobalVocab.counts2vocab(args.counts_file, args.cutoff)
    else:
        sys.exit(0)

    with open(args.output, mode="w", encoding="utf-8") as f:
        json.dump(vocab, f)


# count_days("/home/jakob/lanl/redteam.txt")
# day = 6
# add_redteam_to_log(day, f"/home/jakob/lanl/{day}.csv", f"{day}.csv", "/home/jakob/lanl/redteam.txt", normalized=True)
