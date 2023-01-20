# Kaldi Path stuff:
export PYTHONIOENCODING='utf-8'
export KALDI_ROOT=`pwd`/../kaldi-trunk/
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONUNBUFFERED=1

# Other needed bits:
module purge
export USER_SPACK_ROOT="/projappl/project_2006368/spack_root/"
module load spack
module load gcc/11.3.0 cuda/11.7.0
module load git
module load subversion
module load pytorch/1.12
export PYTHONUSERBASE="/scratch/project_2006368/asr/python-env/"
# varikn installed at:
export PATH=$PATH:/scratch/project_2006368/asr/variKN/build/bin/

source <(spack load --sh sentencepiece)
source <(spack load --sh sox)
source <(spack load --sh flac)
source <(spack load --sh rsync)

export PYTHONPATH=$PYTHONPATH:$PWD/pychain
export PYTHONPATH=$PYTHONPATH:"$PWD/local/python_lib/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:pychain/openfst-1.7.5/lib/

export APPTAINERENV_APPEND_PATH=$PATH
export APPTAINERENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export APPTAINER_BINDPATH="/appl"

