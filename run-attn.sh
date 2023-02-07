#!/usr/bin/env bash
# Attention-model 

stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

# RUN run.sh first, this builds on those shards


if [ $stage -le 1 ]; then
  sbatch local/attention/run_training.sh 
  echo "Submitted training job, exiting"
  exit
fi

if [ $stage -le 2 ]; then
  sbatch local/attention/run_training.sh --hparams hyperparams/attention/CRDNN-AA-contd.yaml
  echo "Submitted training job, exiting"
  exit
fi

