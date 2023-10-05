#!/usr/bin/env bash

stage=1
separate_prior_run=true

. ./cmd.sh
. ./path.sh
. parse_options.sh


# you might not want to do this for interactive shells.
set -e

num_units=$(tree-info exp/chain/tree/tree | grep "num-pdfs" | cut -d" " -f2)
seed=3407

if [ $stage -le 25 ]; then
  ## sbatch won't wait, just exit after sbatching 
  ## and then come back like every three days till its done.
  sbatch local/chain/run_training_parallel_6gpu.sh \
    --py_script "local/chain/sb-train-mtl-conformer.py" \
    --hparams "hyperparams/chain/Conformer-I.yaml"
  exit
fi

if [[ $stage -le 26 && $separate_prior_run == "true" ]]; then
  local/chain/run_training.sh \
    --py_script "local/chain/sb-train-mtl-conformer.py" \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10"
fi


if [ $stage -le 27 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --stage 2 --posteriors_from "exp/chain/Conformer-I/3407-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0" \
    --beam 40 --lattice_beam 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10" \
    --py_script "local/chain/sb-test-conformer-mtl-avg.py" \
    --decodedir "exp/chain/Conformer-I/3407-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0-largebeam"
  local/chain/decode.sh --datadir data/dev_other/ \
    --stage 2 --posteriors_from "exp/chain/Conformer-I/3407-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0" \
    --beam 40 --lattice_beam 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10" \
    --py_script "local/chain/sb-test-conformer-mtl-avg.py" \
    --decodedir "exp/chain/Conformer-I/3407-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0-largebeam"
fi

if [ $stage -le 28 ]; then
  local/chain/decode.sh --datadir data/test_clean \
    --stage 2 --posteriors_from "exp/chain/Conformer-I/3407-${num_units}units/decode_test_clean_bpe.5000.varikn_acwt1.0" \
    --beam 40 --lattice_beam 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10" \
    --py_script "local/chain/sb-test-conformer-mtl-avg.py" \
    --decodedir "exp/chain/Conformer-I/3407-${num_units}units/decode_test_clean_bpe.5000.varikn_acwt1.0-largebeam"
  local/chain/decode.sh --datadir data/test_other/ \
    --stage 2 --posteriors_from "exp/chain/Conformer-I/3407-${num_units}units/decode_test_other_bpe.5000.varikn_acwt1.0" \
    --beam 40 --lattice_beam 20 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10" \
    --py_script "local/chain/sb-test-conformer-mtl-avg.py" \
    --decodedir "exp/chain/Conformer-I/3407-${num_units}units/decode_test_other_bpe.5000.varikn_acwt1.0-largebeam"
fi


## Other LMS:

if [ $stage -le 29 ]; then
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10" \
    --stage 2 --posteriors_from "exp/chain/Conformer-I/3407-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0" \
    --decodedir "exp/chain/Conformer-I/3407-${num_units}units/decode_dev_clean_3gram_pruned_char_acwt1.0" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char" 
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10" \
    --datadir "data/dev_other" \
    --stage 2 --posteriors_from "exp/chain/Conformer-I/3407-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0" \
    --decodedir "exp/chain/Conformer-I/3407-${num_units}units/decode_dev_other_3gram_pruned_char_acwt1.0" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char" 
fi

if [ $stage -le 30 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_clean exp/chain/Conformer-I/3407-${num_units}units/decode_dev_clean_3gram_pruned_char_acwt1.0 \
    exp/chain/Conformer-I/3407-${num_units}units/decode_dev_clean_4gram_char_rescored_acwt1.0
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_other exp/chain/Conformer-I/3407-${num_units}units/decode_dev_other_3gram_pruned_char_acwt1.0 \
    exp/chain/Conformer-I/3407-${num_units}units/decode_dev_other_4gram_char_rescored_acwt1.0
fi


if [ $stage -le 31 ]; then
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10" \
    --stage 2 --posteriors_from "exp/chain/Conformer-I/3407-${num_units}units/decode_test_clean_bpe.5000.varikn_acwt1.0" \
    --decodedir "exp/chain/Conformer-I/3407-${num_units}units/decode_test_clean_3gram_pruned_char_acwt1.0" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char" 
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree \
    --hparams "hyperparams/chain/Conformer-I.yaml --average_n_ckpts 10" \
    --datadir "data/test_other" \
    --stage 2 --posteriors_from "exp/chain/Conformer-I/3407-${num_units}units/decode_test_other_bpe.5000.varikn_acwt1.0" \
    --decodedir "exp/chain/Conformer-I/3407-${num_units}units/decode_test_other_3gram_pruned_char_acwt1.0" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char" 
fi

if [ $stage -le 32 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/test_clean exp/chain/Conformer-I/3407-${num_units}units/decode_test_clean_3gram_pruned_char_acwt1.0 \
    exp/chain/Conformer-I/3407-${num_units}units/decode_test_clean_4gram_char_rescored_acwt1.0
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/test_other exp/chain/Conformer-I/3407-${num_units}units/decode_test_other_3gram_pruned_char_acwt1.0 \
    exp/chain/Conformer-I/3407-${num_units}units/decode_test_other_4gram_char_rescored_acwt1.0
fi

