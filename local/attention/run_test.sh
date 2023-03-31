#!/bin/bash

cmd="slurm.pl --mem 24G --gpu 1 --time 4:0:0"
hparams="hyperparams/attention/CRDNN-E.yaml"
extra_hargs=
py_script="local/attention/sb_test_only_attn.py"
datadir="data/dev_clean"

. path.sh
. parse_options.sh

log=exp/logs/run_test_$(basename $hparams .yaml)_$(basename $datadir)

$cmd $log python $py_script $hparams $extra_hargs --test_data_dir $datadir --test_data_id $(basename $datadir)
