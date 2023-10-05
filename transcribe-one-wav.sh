#!/bin/bash
# This script is more of a debug tool
# It does basic HMM/DNN decoding for a single wavefile
# Use .wav extension (for basename)

hparams="hyperparams/chain/CRDNN-AA-contd.yaml"
treedir="exp/chain/tree"
graphdir="exp/chain/graph/graph_bpe.5000.varikn"
lmwt=13
wip=0.5
frame_shift=0.03
print_silence=false
hyp_filtering_cmd="local/wer_hyp_filter"
ctm_cmd="slurm.pl --mem 2G --time 0:30:0"
nj=1  # Number of jobs
stage=0

. path.sh 
. parse_options.sh

set -eu

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <wavefile> <outputdir>"
  exit 1 
fi

wavefile="$1"
outputdir="$2"
decode_dir="$outputdir"/decode_$(basename "$hparams" .yaml)_$(basename "$graphdir")
data="$outputdir"/datadir
mkdir -p "$data"

if [ $stage -le 0 ]; then
  uttid=$(basename $wavefile .wav)
  echo "$uttid $wavefile" > "$data"/wav.scp
  echo "$uttid $uttid" > "$data"/utt2spk
  echo "$uttid $uttid" > "$data"/spk2utt
fi

if [ $stage -le 1 ]; then
  local/chain/decode.sh --datadir "$data" \
    --acwt 1.5 --post-decode-acwt 15.0 \
    --hparams "$hparams" \
    --graphdir "$graphdir" \
    --tree "$treedir" \
    --nj $nj \
    --skip_scoring "true" \
    --decode_cmd "slurm.pl --mem 12G --time 0:30:0" \
    --am_cmd "srun --gres=gpu:v100:1 --time 0:10:0 --mem 32G --partition gpu --account project_2006368" \
    --decodedir "$decode_dir"
fi

# This section instead just gets the transcript
#
if [ $stage -le 2 ]; then
  # This section copies steps/score_kaldi.sh
  symtab="$graphdir"/words.txt
  model="$treedir"/final.mdl
  nj=$(cat "$decode_dir"/num_jobs)

  $ctm_cmd JOB=1:$nj "$outputdir"/log/best_path_${lmwt}_${wip}.JOB.log \
    lattice-scale --inv-acoustic-scale=$lmwt "ark:gunzip -c $decode_dir/lat.JOB.gz|" ark:- \| \
    lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
    lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
    utils/int2sym.pl -f 2- $symtab \| \
    $hyp_filtering_cmd '>' "$outputdir"/${lmwt}_${wip}.JOB.txt || exit 1;

  for n in `seq $nj`; do 
    cat "$outputdir"/${lmwt}_${wip}.$n.txt
  done > "$outputdir"/${lmwt}_${wip}.txt
  ln -s "$PWD"/"$outputdir"/${lmwt}_${wip}.txt "$outputdir"/transcripts.txt

  for n in `seq $nj`; do 
    rm "$outputdir"/${lmwt}_${wip}.$n.txt
  done
fi

