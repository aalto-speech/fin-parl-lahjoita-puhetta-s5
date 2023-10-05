#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=15:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --account=project_2006368
#SBATCH --cpus-per-task=2

hparams="hyperparams/fusion/CRDNN-AA-contd_Trafo-A.yaml"
py_script="local/fusion/sb-test-shallow-fusion.py"
test_data_dir="data/dev_all/"

. path.sh
. parse_options.sh

test_data_id=$(basename $test_data_dir)

python $py_script $hparams --test_data_dir "$test_data_dir" --test_data_id "$test_data_id"

