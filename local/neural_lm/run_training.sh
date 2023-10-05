#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --account=project_2006368
#SBATCH --cpus-per-task=2

hparams="hyperparams/lm/Trafo-Everything-A.yaml"
py_script="local/neural_lm/sb-train-trafo-lm.py"
num_units=5000
seed=2224

. path.sh
. parse_options.sh

timesfailed=0
while ! python $py_script $hparams --num_units "$num_units" --seed "$seed"; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 10 ]; then
    echo "Training crashed, restarting!"
    sleep 3
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

