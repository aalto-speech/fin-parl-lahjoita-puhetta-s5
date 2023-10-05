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

if [ $stage -le 3 ]; then
  sbatch local/attention/run_training.sh --py_script local/attention/sb_train_attn_mwer.py --hparams hyperparams/attention/mwer/CRDNN-AA-contd.yaml
  echo "Submitted training job, exiting"
  exit
fi

if [ $stage -le 4 ]; then
  local/attention/run_test.sh --hparams hyperparams/attention/CRDNN-AA-contd.yaml --datadir data/lp-dev
  local/attention/run_test.sh --hparams hyperparams/attention/CRDNN-AA-contd.yaml --datadir data/parl-dev-all
  local/attention/run_test.sh --hparams hyperparams/attention/CRDNN-AA-contd.yaml --datadir data/parl-test-all
  local/attention/run_test.sh --hparams hyperparams/attention/CRDNN-AA-contd.yaml --datadir data/parl-test-2020
  local/attention/run_test.sh --hparams hyperparams/attention/CRDNN-AA-contd.yaml --datadir data/lp-test
fi

if [ $stage -le 5 ]; then
  # Note: this data set is not available publicly.
  local/attention/run_test.sh --hparams hyperparams/attention/CRDNN-AA-contd.yaml --datadir data/yle-test-new
fi

# Fix the input dimensionality
if [ $stage -le 6 ]; then
  sbatch local/attention/run_training.sh  \
    --hparams "hyperparams/attention/FIX-CRDNN-aa.yaml"
  echo "Submitted training job, exiting"
  exit
fi

if [ $stage -le 7 ]; then
  sbatch local/attention/run_training.sh --hparams hyperparams/attention/FIX-CRDNN-aa-contd.yaml
  echo "Submitted training job, exiting"
  exit
fi
