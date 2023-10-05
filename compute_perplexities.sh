#!/bin/bash

models="exp/lm/train_varikn.bpe5000.d0.0001/ exp/lm/train_parl_varikn.bpe5000.d0.0001/ exp/lm/train_lp_varikn.bpe5000.d0.0001/"
#datas="parl-dev-all parl-test-all parl-test-2020 lp-dev lp-test yle-test-new"
datas="lp-dev lp-test"



for model in $models; do
  modelname=$(basename $model)
  mkdir -p data/processed-lm-data/$modelname
  for data in $datas; do
    spm_encode --model $model/bpe.5000.model <(cut -d" " -f2- data/$data/text | python local/preprocess_lm_data.py) | sed "s:$: </s>:g" \
      > data/processed-lm-data/$modelname/$data
    perplexity --arpa $model/varikn.lm.gz data/processed-lm-data/$modelname/$data $model/perplexity_$data
  done
done
