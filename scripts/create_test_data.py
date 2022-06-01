first_events = {6: 5981956, 7: 1877719, 8: 2518706}

interval = 100

for day, line_num in first_events.items():
    with open(f"data/test_data/{day}.csv", "w") as outfile, open(f"data/full_data/{day}.csv", "r") as infile:
        for i, line in enumerate(infile):
            if i >= line_num + interval:
                continue
            if i >= line_num - interval:
                outfile.write(line)
            else:
                continue
