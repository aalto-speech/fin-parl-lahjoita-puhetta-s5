#!/bin/bash

dataroot="/scratch/project_2006368/asr/dataroot/eduskunta-new/fi-parliament-asr/"


# dev-seen:
dirin=$dataroot/dev-test/2016-dev/seen
outdir="data/parl-dev-seen"
mkdir -p $outdir
while read fullwavpath; do
  uttid=$(basename $fullwavpath .wav)
  spkid=${uttid%_*}

  fulltrnpath=${fullwavpath%.wav}.trn

  echo "$uttid $fullwavpath" >> $outdir/wav.scp
  cat <(echo -n "$uttid ") $fulltrnpath >> $outdir/text
  echo "$uttid $spkid" >> $outdir/utt2spk
done < <(find "$dirin" -iname "*.wav")

# dev-unseen:
dirin=$dataroot/dev-test/2016-dev/unseen
outdir="data/parl-dev-unseen"
mkdir -p $outdir
while read fullwavpath; do
  uttid=$(basename $fullwavpath .wav)
  spkid=${uttid%_*}

  fulltrnpath=${fullwavpath%.wav}.trn

  echo "$uttid $fullwavpath" >> $outdir/wav.scp
  cat <(echo -n "$uttid ") $fulltrnpath >> $outdir/text
  echo "$uttid $spkid" >> $outdir/utt2spk
done < <(find "$dirin" -iname "*.wav")
# FIX one utterance:
sed -i "s/ymmärtä'/ymmärtää/g" data/parl-dev-unseen/text

# test-seen:
dirin=$dataroot/dev-test/2016-test/seen
outdir="data/parl-test-seen"
mkdir -p $outdir
while read fullwavpath; do
  uttid=$(basename $fullwavpath .wav)
  spkid=${uttid%_*}

  fulltrnpath=${fullwavpath%.wav}.trn

  echo "$uttid $fullwavpath" >> $outdir/wav.scp
  cat <(echo -n "$uttid ") $fulltrnpath >> $outdir/text
  echo "$uttid $spkid" >> $outdir/utt2spk
done < <(find "$dirin" -iname "*.wav")

# test-unseen:
dirin=$dataroot/dev-test/2016-test/unseen
outdir="data/parl-test-unseen"
mkdir -p $outdir
while read fullwavpath; do
  uttid=$(basename $fullwavpath .wav)
  spkid=${uttid%_*}

  fulltrnpath=${fullwavpath%.wav}.trn

  echo "$uttid $fullwavpath" >> $outdir/wav.scp
  cat <(echo -n "$uttid ") $fulltrnpath >> $outdir/text
  echo "$uttid $spkid" >> $outdir/utt2spk
done < <(find "$dirin" -iname "*.wav")

# test-2020:
dirin=$dataroot/dev-test/2020-test
outdir="data/parl-test-2020"
mkdir -p $outdir
while read fullwavpath; do
  uttid=$(basename $fullwavpath .wav)
  spkid=$(echo $uttid |cut -d"-" -f1)

  fulltrnpath=${fullwavpath%.wav}.trn

  echo "$uttid $fullwavpath" >> $outdir/wav.scp
  cat <(echo -n "$uttid ") $fulltrnpath >> $outdir/text
  echo "$uttid $spkid" >> $outdir/utt2spk
done < <(find "$dirin" -iname "*.wav")
