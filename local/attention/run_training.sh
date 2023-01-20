#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --account=project_2006368
#SBATCH --cpus-per-task=5

hparams="hyperparams/attention/CRDNN-AA.yaml"
py_script="local/attention/sb_train_attn.py"

. path.sh
. parse_options.sh

timesfailed=0
while ! python $py_script $hparams; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 5 ]; then
    echo "Training crashed, restarting!"
    sleep 3
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

