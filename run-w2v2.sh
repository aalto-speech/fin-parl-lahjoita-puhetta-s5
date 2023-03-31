#!/usr/bin/env bash
# Wav2vec 2.0 Chain style model 

lp_dataroot="/scratch/project_2006368/asr/dataroot/puhelahjat/puhelahjat/"
mfccdir=mfcc
stage=22

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e


# RUN run.sh first, this builds on that.

if [ $stage -le 22 ]; then
  local/chain/build_new_tree.sh \
    --frame_subsampling_factor 2 \
    --traindata data/train_all \
    --trainali exp/tri6b_ali_all \
    --validali exp/tri6b_ali_dev_2k \
    --num_leaves 4000 \
    exp/chain/tree2
fi

if [ $stage -le 23 ]; then
  srun --mem 24G --time 2-0:0:0 -c 20 --account project_2006368 \
    local/chain/make_shards.py 180 shards/train_all_sub2 \
      --num-proc 16 \
      --segments data/train_all/split180/JOB/segments \
                 data/train_all/split180/JOB/wav.scp \
      --text data/train_all/split180/JOB/text \
      --aliark "gunzip -c exp/chain/tree2/ali.JOB.gz | ali-to-pdf exp/chain/tree2/final.mdl ark:- ark:- |"

  srun --mem 6G --time 12:0:0 -c 2 --account project_2006368 \
    local/chain/make_shards.py 8 shards/dev_2k_sub2 \
      --num-proc 2 \
      --segments data/dev_2k/split8/JOB/segments \
                 data/dev_2k/split8/JOB/wav.scp \
      --text data/dev_2k/split8/JOB/text \
      --aliark "gunzip -c exp/chain/tree2/ali.valid.JOB.gz | ali-to-pdf exp/chain/tree2/final.mdl ark:- ark:- |"
fi

if [ $stage -le 24 ]; then
  local/chain/prepare_graph_clustered.sh \
    --treedir exp/chain/tree2 \
    --trainset train_all \
    --validset dev_2k \
    --graph exp/chain/graph2
fi

if [ $stage -le 25 ]; then
  sbatch local/chain/run_training.sh \
    --py_script local/chain/sb-train-mtl-w2v2.py \
    --treedir exp/chain/tree2 \
    --hparams hyperparams/chain/W2V2-A.yaml
  echo "Sent the neural network training into queue, exiting run script"
  exit 
fi

if [ $stage -le 26 ]; then
  local/chain/decode.sh --datadir data/dev_all/ \
    --nj 24 \
    --py_script local/chain/sb-test-w2v2-mtl-avg.py \
    --tree exp/chain/tree2 \
    --acwt 1.5 --post-decode-acwt 15.0 \
    --hparams "hyperparams/chain/W2V2-A.yaml --tmpstorage ./tmp/" \
    --decodedir "exp/chain/W2V2-A/2602-3096units/decode_dev_all_bpe.5000.varikn_acwt1.5"
fi
