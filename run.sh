#!/usr/bin/env bash

lp_dataroot="/scratch/project_2006368/asr/dataroot/puhelahjat/puhelahjat/"
mfccdir=mfcc
stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e


# TODO: ADD DATA PREP HERE, from local/

if [ $stage -le 0 ]; then
  local/textgrid_to_kaldi.py \
    $lp_dataroot'/v1/*/alignments/*/*.TextGrid' \
    data/all_aligned/
  # Fix some utterances:
  sed -i "s/\+//g" data/all_aligned/text
  sed -i "s/hafþór/haftor/g" data/all_aligned/text
  sed -i "s/ëttä/että/g" data/all_aligned/text
  local/list-puhelahjat.sh
  local/filter-dev-test.sh \
    data/lp-dev-test-utt2spks/lp-test-utt2spk \
    data/all_aligned \
    data/lp-test
  local/filter-dev-test.sh \
    data/lp-dev-test-utt2spks/lp-dev-utt2spk \
    data/all_aligned \
    data/lp-dev
  local/filter-dev-test.sh \
    data/lp-dev-test-utt2spks/lp-test-multitranscriber-utt2spk \
    data/all_aligned \
    data/lp-test-multitranscriber
  local/filter-dev-test.sh \
    data/lp-dev-test-utt2spks/lp-test-multitranscriber-speakers-utt2spk \
    data/all_aligned \
    data/lp-test-multitranscriber-speakers
  local/prep_eduskunta.sh
  local/prep_eduskunta_dev_test.sh
fi

if [ $stage -le 1 ]; then
  tmpdir=$(mktemp -d)
  cat data/lp-dev/utt2spk data/lp-test/utt2spk \
    data/lp-test-multitranscriber/utt2spk \
    data/lp-test-multitranscriber-speakers/utt2spk \
    > "$tmpdir"/all-lp-dev-test-utt2spk
  utils/filter_scp.pl --exclude "$tmpdir"/all-lp-dev-test-utt2spk \
    < data/all_aligned/utt2spk \
    > "$tmpdir"/all-lp-train-utt2spk
  utils/subset_data_dir.sh --utt-list "$tmpdir"/all-lp-train-utt2spk \
    data/all_aligned/ data/lp-train-all
  rm -rf "$tmpdir"
fi

if [ $stage -le 2 ]; then
  for utt2spk in data/*/utt2spk; do
    datadir=$(dirname $utt2spk)
    utils/fix_data_dir.sh $datadir
  done
fi

if [ $stage -le 3 ]; then
  for part in lp-test lp-dev lp-train-all parl-2008-2020-train; do
    steps/make_mfcc.sh --cmd "$train_cmd --time 4:0:0" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
fi

if [ $stage -le 4 ]; then
  utils/combine_data.sh \
    data/train_all data/lp-train-all data/parl-2008-2020-train
fi

if [ $stage -le 5 ]; then
  # Make some small data subsets for early system-build stages.

  utils/subset_data_dir.sh --shortest data/train_all 4000 data/train_4kshort
  utils/subset_data_dir.sh data/train_all 10000 data/train_10k
  utils/subset_data_dir.sh data/train_all 20000 data/train_20k
  utils/subset_data_dir.sh data/train_all 400000 data/train_400k
  utils/subset_data_dir.sh data/train_all 2000000 data/train_2M
  for dataname in train_4kshort train_10k train_20k train_400k train_2M; do
    for letter in b c d f g q w x z å ".fp" ".br"; do
      local/enforce_letter_in_data.sh --max_utt_to_add 10 data/train_all/ $letter data/$dataname
    done
  done
fi

if [ $stage -le 6 ]; then
  local/prepare_lexicon.sh \
    --extra_texts "data/lp-dev/text data/parl-dev-seen/text data/parl-dev-unseen/text" \
    data/train_all/ data/local/dict_train_all data/lang_train
fi

#if [ $stage -le 7 ]; then
#  if [ ! -d subword-kaldi ]; then
#    echo "Need subword-kaldi, cloning"
#    git clone https://github.com/aalto-speech/subword-kaldi
#  fi
#
#  local/train_lm.sh \
#    --BPE_units 5000 \
#    --stage 0 \
#    --traindata data/train_960 \
#    --validdata data/dev_all \
#    train data/lang_bpe.5000.varikn
#fi

if [ $stage -le 8 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
                      data/train_4kshort data/lang_train exp/mono
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
                    data/train_10k data/lang_train exp/mono exp/mono_ali_10k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_10k data/lang_train exp/mono_ali_10k exp/tri1
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
                    data/train_20k data/lang_train exp/tri1 exp/tri1_ali_20k


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_20k data/lang_train exp/tri1_ali_20k exp/tri2b
fi

if [ $stage -le 11 ]; then
  # Align a 20k utts subset using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
                     data/train_20k data/lang_train exp/tri2b exp/tri2b_ali_20k

  # Train tri3b, which is LDA+MLLT+SAT on 20k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_20k data/lang_train exp/tri2b_ali_20k exp/tri3b

fi

if [ $stage -le 12 ]; then
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train_400k data/lang_train \
    exp/tri3b exp/tri3b_ali_400k

  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/train_400k data/lang_train \
                      exp/tri3b_ali_400k exp/tri4b
fi

if [ $stage -le 16 ]; then
  steps/align_fmllr.sh --nj 120 --cmd "$train_cmd --time 4:0:0" \
                       data/train_2M data/lang_train exp/tri4b exp/tri4b_ali_2M

  steps/train_sat.sh  --cmd "$train_cmd" 10000 200000 \
                      data/train_2M data/lang_train exp/tri4b_ali_2M exp/tri5b
fi

if [ $stage -le 18 ]; then
  # NOTE: on CSC hardware, splitting train_all into 180 pieces
  #  results in "Out of memory!".
  #  I ran it manually with: 
  #  > source path.sh
  #  > srun --mem 20G --gres=nvme:20 --time 1:0:0 -c4 \
  #  >   --account project_2006368 split_data.sh data/train_all 180
  steps/align_fmllr.sh --nj 180 --cmd "$train_cmd --time 4:0:0" \
                       data/train_all data/lang_train exp/tri5b exp/tri5b_ali_all

  steps/train_quick.sh --cmd "$train_cmd" \
                       12000 300000 data/train_all data/lang_train exp/tri5b_ali_all exp/tri6b

fi

exit

if [ $stage -le 19 ]; then
  $basic_cmd --mem 16G exp/tri6b/graph_bpe.5000.varikn/log/mkgraph.log utils/mkgraph.sh \
    data/lang_bpe.5000.varikn/ exp/tri6b/ exp/tri6b/graph_bpe.5000.varikn
  steps/decode_fmllr.sh --cmd "$basic_cmd" --nj 8 \
    exp/tri6b/graph_bpe.5000.varikn data/dev_clean exp/tri6b/decode_dev_clean_bpe.5000.varikn
  steps/decode_fmllr.sh --cmd "$basic_cmd" --nj 8 \
    exp/tri6b/graph_bpe.5000.varikn data/dev_other exp/tri6b/decode_dev_other_bpe.5000.varikn
fi

if [ $stage -le 20 ]; then
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/train_all data/lang_train exp/tri6b exp/tri6b_ali_all
  steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
    data/dev_all data/lang_train exp/tri6b exp/tri6b_ali_dev_all
fi

if [ $stage -le 21 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang_train/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang_train..."
      echo " ... not sure what to do. Exiting."
      exit 1;
    fi
  else
    cp -r data/lang_train data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 22 ]; then
  local/chain/build_new_tree.sh exp/chain/tree
fi

if [ $stage -le 23 ]; then
  srun --mem 24G --time 1-12:0:0 -c8 \
    local/chain/make_shards.py 100 shards/train_all \
      --num-proc 8 \
      --wavscp data/train_all/split100/JOB/wav.scp \
      --text data/train_all/split100/JOB/text \
      --aliark "gunzip -c exp/chain/tree/ali.JOB.gz | ali-to-pdf exp/chain/tree/final.mdl ark:- ark:- |"

  srun --mem 6G --time 12:0:0 -c2 \
    local/chain/make_shards.py 8 shards/dev_all \
      --num-proc 2 \
      --wavscp data/dev_all/split8/JOB/wav.scp \
      --text data/dev_all/split8/JOB/text \
      --aliark "gunzip -c exp/chain/tree/ali.valid.JOB.gz | ali-to-pdf exp/chain/tree/final.mdl ark:- ark:- |"
fi

if [ $stage -le 24 ]; then
  local/chain/prepare_graph_clustered.sh
fi

if [ $stage -le 25 ]; then
  local/chain/run_training.sh
fi

if [ $stage -le 26 ]; then
  $basic_cmd --mem 16G exp/chain/graph/graph_bpe.5000.varikn/log/mkgraph.log utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_bpe.5000.varikn/ exp/chain/tree exp/chain/graph/graph_bpe.5000.varikn
fi

if [ $stage -le 27 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 28 ]; then
  local/chain/run_training.sh \
    --hparams "hyperparams/chain/New-CRDNN-J-contd.yaml"
fi

if [ $stage -le 29 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --hparams "hyperparams/chain/New-CRDNN-J-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J-contd/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --hparams "hyperparams/chain/New-CRDNN-J-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J-contd/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi
