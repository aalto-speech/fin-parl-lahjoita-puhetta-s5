#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --account=project_2006368
#SBATCH --cpus-per-task=5

hparams="hyperparams/chain/CRDNN-AA.yaml"
treedir="exp/chain/tree/"
py_script="local/chain/sb-train-mtl.py"

. path.sh
. parse_options.sh

num_units=$(tree-info $treedir/tree | grep "num-pdfs" | cut -d" " -f2)

timesfailed=0
while ! python $py_script $hparams --num_units $num_units --tmpstorage $LOCAL_SCRATCH; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 100 ]; then
    echo "Training crashed, restarting!"
    sleep 3
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

