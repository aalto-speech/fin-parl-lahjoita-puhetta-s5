#!/bin/bash

cmd="srun --mem 24G --time 1:0:0 -c5 --gres=gpu:1 --constraint volta -p dgx-spa,gpu,gpu-nvlink,gpushort"
hparams="hyperparams/attention/CRDNN-E.yaml"
py_script="local/attention/sb_test_only_attn.py"
datadir="data/dev_clean"

. path.sh
. parse_options.sh

$cmd python $py_script $hparams --test_data_dir $datadir --test_data_id $(basename $datadir)
