# Kaldi Path stuff:
export KALDI_ROOT=`pwd`/../kaldi-trunk/
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONUNBUFFERED=1

# Other needed bits:
module load python-data
export PYTHONUSERBASE="/scratch/project_2006368/asr/python-env/"
