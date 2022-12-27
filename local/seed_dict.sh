#!/usr/bin/env bash

# Copyright 2022 Anja Virkkunen, Aku Rouhe
# Apache 2.0

if [ $# -ne 1 ]; then
    echo "Usage: local/seed_dict.sh <lang_tmp>"
    echo "e.g.: $0 data/local/lang"
    echo
    echo "Create files silence_phones, optional_silence, and extra_questions."
    exit 1
fi

tmpdir=$1

. path.sh

# These should just match your silence phones:
echo "SIL" >${tmpdir}/silence_phones.txt
echo "SPN" >>${tmpdir}/silence_phones.txt
echo "NSN" >> ${tmpdir}/silence_phones.txt
sort -uo ${tmpdir}/silence_phones.txt{,}

echo "SIL" >${tmpdir}/optional_silence.txt

echo ".fp SPN" > ${tmpdir}/lexicon.txt
echo ".ct NSN" >> ${tmpdir}/lexicon.txt
echo ".cough NSN" >> ${tmpdir}/lexicon.txt
echo ".laugh NSN" >> ${tmpdir}/lexicon.txt
echo ".yawn NSN" >> ${tmpdir}/lexicon.txt
echo ".sigh NSN" >> ${tmpdir}/lexicon.txt
echo ".br NSN" >> ${tmpdir}/lexicon.txt
echo "<UNK> SPN" >> ${tmpdir}/lexicon.txt

touch $tmpdir/extra_questions.txt
