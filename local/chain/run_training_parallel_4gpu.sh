#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4,nvme:60
#SBATCH --account=project_2006368
#SBATCH --cpus-per-task=20

hparams="hyperparams/chain/Conformer-I.yaml"
treedir="exp/chain/tree/"
py_script="local/chain/sb-train-mtl-conformer.py"
num_proc=4
master_port=47780  # NOTE! on the same machine, you have to use your own port
# Should upgrade to torch.distributed.run

. path.sh
. parse_options.sh

num_units=$(tree-info $treedir/tree | grep "num-pdfs" | cut -d" " -f2)

timesfailed=0
while ! python -m torch.distributed.launch --nproc_per_node=$num_proc --master_port $master_port $py_script --distributed_launch --find_unused_parameters $hparams --num_units $num_units --tmpstorage $LOCAL_SCRATCH; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 100 ]; then
    echo "Training crashed, restarting!"
    # Kill all dangling processes:
    killall --user rouheaku python
    sleep 10
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

