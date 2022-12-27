#!/usr/bin/env python3
import sys
import re
pattern = re.compile("([0-9]+)")

def convert_uttid(uttid):
    parts = []
    first_found = False
    last_was_number = False
    for part in pattern.split(uttid):
        if str.isnumeric(part):
            if not first_found:
                parts.append(part.zfill(7))
                first_found = True
            else:
                parts.append(part)
            last_was_number = True
        elif str.isalpha(part):
            if last_was_number:
                parts.append("_")
            parts.append(part)
            last_was_number = False
    return "".join(parts)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    with open(args.file) as fin:
        for line in fin:
            uttid, *_ = line.strip().split(maxsplit=1)
            new_uttid = convert_uttid(uttid)
            print(new_uttid)
