#!/bin/bash
set -eu
. cmd.sh

stage=1
num_units=5000
seed=2224
lmdatadir="data/lmdata_everything"
lmdir="exp/lm/everything_trafo-A/bpe${num_units}-seed${seed}"
traindata="kielipankki.txt parl_30M.txt parl_and_lp.txt webcon_and_dsp.txt"
validdata="yle_dev.txt parl_and_lp_dev.txt"
hparams="hyperparams/lm/Trafo-Everything-A.yaml"

echo $0 $@

. path.sh
. parse_options.sh


if [ "$stage" -le 0 ]; then
  echo "LM training data prep"
  mkdir -p "$lmdatadir"
  # NOTE: Need to remove first column from the text, as that is the Kaldi uttid
  rm -f "$lmdatadir"/plain_text
  for textfile in $traindata; do
    local/preprocess_lm_data.py <$lmdatadir/$textfile >> "$lmdatadir"/plain_text
  done
  # Deduplicate and shuffle:
  sort --unique --random-sort --output "$lmdatadir"/plain_text "$lmdatadir"/plain_text
  echo "Training data done, now LM dev data"
  rm -f "$lmdatadir"/plain_text.valid
  for textfile in $validdata; do
    local/preprocess_lm_data.py <$lmdatadir/$textfile >> "$lmdatadir"/plain_text.valid
  done
fi

exit # Make sure everything is fine.

if [ $stage -le 1 ]; then
  sbatch local/neural_lm/run_training.sh \
    --hparams hyperparams/lm/Trafo-Everything-A.yaml \
    --seed $seed \
    --num_units $num_units
fi
