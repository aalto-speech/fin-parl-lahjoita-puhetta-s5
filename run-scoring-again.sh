#!/bin/bash
# rerun the scoring for LP results!
#SBATCH --mem=24G
#SBATCH --time=1:0:0
#SBATCH --cpus-per-task=8
#SBATCH --account=project_2006368

#local/score.sh data/lp-dev/ exp/chain/graph/graph_bpe.5000.varikn/ exp/chain/CRDNN-AA-contd/2602-2856units/decode_lp-dev_bpe.5000.varikn_acwt1.5/
#local/score.sh data/lp-test exp/chain/graph/graph_bpe.5000.varikn/ exp/chain/CRDNN-AA-contd/2602-2856units/decode_lp-test_bpe.5000.varikn_acwt1.5/

local/score.sh --min_lmwt 3 --max_lmwt 7 data/lp-dev/ exp/chain/graph/graph_parl_bpe.5000.varikn/ exp/chain/CRDNN-AA-contd/2602-2856units/decode_lp-dev_parl_bpe.5000.varikn_acwt1.5/
local/score.sh --min_lmwt 3 --max_lmwt 7 data/lp-test exp/chain/graph/graph_parl_bpe.5000.varikn/ exp/chain/CRDNN-AA-contd/2602-2856units/decode_lp-test_parl_bpe.5000.varikn_acwt1.5/

#local/score.sh data/lp-dev/ exp/chain/graph/graph_lp_bpe.5000.varikn/ exp/chain/CRDNN-AA-contd/2602-2856units/decode_lp-dev_lp_bpe.5000.varikn_acwt1.5/
#local/score.sh data/lp-test exp/chain/graph/graph_lp_bpe.5000.varikn/ exp/chain/CRDNN-AA-contd/2602-2856units/decode_lp-test_lp_bpe.5000.varikn_acwt1.5/


