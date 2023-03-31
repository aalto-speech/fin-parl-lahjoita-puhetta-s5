#!/bin/bash
# This script does basic decoding with an HMM/DNN ASR system.
# This is designed to take in a Kaldi data directory produced with
# NeMo and the connected scripts. 
# Then, after decoding, this script will produce a CTM file
# which has timestamps for each output word.

hparams="hyperparams/chain/CRDNN-AA-contd.yaml"
treedir="exp/chain/tree"
graphdir="exp/chain/graph/graph_bpe.5000.varikn"
lmwt=13
wip=0.5
frame_shift=0.03
print_silence=false
hyp_filtering_cmd="local/wer_hyp_filter"
ctm_cmd="slurm.pl --mem 2G --time 0:30:0"
nj=128  # Number of jobs
stage=0

. path.sh 
. parse_options.sh

set -eu

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <datadir> <outputdir>"
  exit 1 
fi

data="$1"
outputdir="$2"
decode_dir="$outputdir"/decode_$(basename "$hparams" .yaml)_$(basename "$graphdir")

if [ $stage -le 0 ]; then
  # Note: the incoming datadir may not have uttids which work for kaldi's
  # sorting. This doesn't really matter for our decoding runs though.
  # Thus we don't run "fix"
  #utils/fix_data_dir.sh "$data"

  utils/utt2spk_to_spk2utt.pl "$data"/utt2spk > "$data"/spk2utt
  mkdir -p "$decode_dir"
fi

if [ $stage -le 1 ]; then
  # The decoding time should be less than 4 hours (times number of jobs)
  # However, the acoustic model decodes everything sequentially and runs on the GPU
  # It will probably take around 2 hours, but I suggest reserving for 4.
  local/chain/decode.sh --datadir "$data" \
    --acwt 1.5 --post-decode-acwt 15.0 \
    --hparams "$hparams" \
    --graphdir "$graphdir" \
    --tree "$treedir" \
    --nj $nj \
    --skip_scoring "true" \
    --decode_cmd "slurm.pl --mem 12G --time 4:0:0" \
    --am_cmd "srun --gres=gpu:v100:1 --time 4:0:0 --mem 32G --partition gpu --account project_2006368" \
    --decodedir "$decode_dir"
fi

# This section gets the CTM
# However, it doesn't work with the current Kaldi Subword Lang directory 
# (see aalto-speech/subword-kaldi note on GitHub)
#
#if [ $stage -le 2 ]; then
#  # NOTE: this steps copies steps/get_ctm_fast.sh
#  # but just fixes some of the Kaldi assumptions (like having final.mdl in dir above decode dir)
#  model="$treedir"/final.mdl
#  nj=$(cat "$decode_dir"/num_jobs)
#
#  if [ -f "$graphdir"/phones/word_boundary.int ]; then
#    $ctm_cmd JOB=1:$nj $outputdir/log/get_ctm.JOB.log \
#      set -o pipefail '&&' \
#      lattice-1best --lm-scale=$lmwt --word-ins-penalty=$wip "ark:gunzip -c $decode_dir/lat.JOB.gz|" ark:- \| \
#      lattice-align-words "$graphdir"/phones/word_boundary.int $model ark:- ark:- \| \
#      nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - \| \
#      utils/int2sym.pl -f 5 "$graphdir"/words.txt \
#      '>' "$outputdir"/ctm.JOB || exit 1;
#  elif [ -f "$graphdir"/phones/align_lexicon.int ]; then
#    $ctm_cmd JOB=1:$nj "$outputdir"/log/get_ctm.JOB.log \
#      set -o pipefail '&&' \
#      lattice-1best --lm-scale=$lmwt --word-ins-penalty=$wip "ark:gunzip -c $decode_dir/lat.JOB.gz|" ark:- \| \
#      lattice-align-words-lexicon "$graphdir"/phones/align_lexicon.int $model ark:- ark:- \| \
#      lattice-1best ark:- ark:- \| \
#      nbest-to-ctm --frame-shift=$frame_shift --print-silence=$print_silence ark:- - \| \
#      utils/int2sym.pl -f 5 "$graphdir"/words.txt \
#      '>' "$outputdir"/ctm.JOB || exit 1;
#  else
#    echo "$0: neither "$graphdir"/phones/word_boundary.int nor "$graphdir"/phones/align_lexicon.int exists: cannot align."
#    exit 1;
#  fi
#
#  for n in `seq $nj`; do 
#    cat "$outputdir"/ctm.$n
#    rm "$outputdir"/ctm.$n
#  done > "$outputdir"/ctm
#fi

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

