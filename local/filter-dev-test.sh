#!/bin/bash

. path.sh

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <utt2spk-file> <data-in> <data-out>"
  exit 1
fi

utt2spk_orig="$1"
datain="$2"
dataout="$3"

tmpdir=$(mktemp -d)

python local/old_uttid_to_new.py "$utt2spk_orig" > "$tmpdir"/recid_filter
utils/filter_scp.pl -f 2 "$tmpdir"/recid_filter <"$datain"/segments > "$tmpdir"/uttid_filter

utils/subset_data_dir.sh --utt-list "$tmpdir"/uttid_filter "$datain" "$dataout"

rm -rf "$tmpdir"

