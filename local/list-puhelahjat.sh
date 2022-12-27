#!/bin/bash

dataroot="/scratch/project_2006368/asr/dataroot/puhelahjat/puhelahjat"

aligned="data/all_aligned"
justannotations="data/just_annotation"
other="data/other"

mkdir -p "$aligned"
mkdir -p "$justannotations"
mkdir -p "$other"

for audiofile in "$dataroot"/v1/*/audio/*.flac; do
  namestem=$(basename "$audiofile" .flac)
  possible_ali=$(dirname "$audiofile")/../alignments/manual_*/"$namestem".TextGrid
  possible_annotation=$(dirname "$audiofile")/../annotations/manual_*/"$namestem".txt
  spkid=$(cut -d"_" -f1 <(echo "$namestem"))
  if [ -f $possible_ali ]; then
    # JUST MAKE wav.scp here, textgrid_to_kaldi.py takes care of the rest
    echo "$namestem flac -c -d -s $audiofile | sox -t wav - -r 16000 -t wav - |" >> "$aligned/wav.scp"
  else 
    if [ -f $possible_annotation ]; then
      echo "$namestem flac -c -d -s $audiofile | sox -t wav - -r 16000 -t wav - |" >> "$justannotations"/wav.scp
      cat <(echo -n "$namestem ") <(sed "2,\$s/^/.pause /g" $possible_annotation) >> "$justannotations"/text
      echo "$namestem $spkid" >> "$justannotations"/utt2spk
    else # No alignment and no annotation:
      echo "$namestem flac -c -d -s $audiofile | sox -t wav - -r 16000 -t wav - |" >> "$other"/wav.scp
      echo "$namestem $spkid" >> "$other"/utt2spk
    fi
  fi
done

