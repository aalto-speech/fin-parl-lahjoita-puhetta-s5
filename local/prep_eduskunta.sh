#!/bin/bash

dataroot="/scratch/project_2006368/asr/dataroot/eduskunta-new/fi-parliament-asr/"
oldroot="$dataroot"/2008-2016set/
newroot="$dataroot"/2015-2020set/

outdir="data/parl-2008-2020-train"
mkdir -p $outdir


#2008-2014
while read wavpath; do
  uttid=$(basename $wavpath .wav | sed "s/\.//g")
  spkid=$(echo $uttid | sed -r "s/_[0-9]+//g")

  fullwavpath="$oldroot"/$wavpath
  fulltrnpath="$oldroot"/${wavpath%.wav}.trn

  echo "$uttid $fullwavpath" >> $outdir/wav.scp
  cat <(echo -n "$uttid ") $fulltrnpath >> $outdir/text
  echo "$uttid $spkid" >> $outdir/utt2spk
done < "$oldroot"/2008-2014-samples.list
echo "Done 2008-2014"
date

#2015-2020
while read fullwavpath; do
  uttid=$(basename $fullwavpath .wav)
  spkid=$(echo $uttid |cut -d"-" -f1)

  fulltrnpath=${fullwavpath%.wav}.trn

  echo "$uttid $fullwavpath" >> $outdir/wav.scp
  cat <(echo -n "$uttid ") $fulltrnpath >> $outdir/text
  echo "$uttid $spkid" >> $outdir/utt2spk
done < <(find "$newroot" -iname "*.wav")
echo "Done 2015-2020"

# FIX one utterance:
sed -i "s/viejÃ€Ã€/viejää/g" $outdir/text
