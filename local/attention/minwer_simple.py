#!/usr/bin/env python3 
# This file is copied from work done in SpeechBrain,
# see the comment that follows.
# This implements minimum WER training
"""
minWER loss implementation
and based on https://arxiv.org/pdf/1712.01818.pdf
Authors
 * Aku Rouhe 2022
 * Sung-Lin Yeh 2020
 * Abdelwahab Heba 2020
"""
import torch

def minWER_loss_given(
    wers,
    hypotheses_scores,
    subtract_avg=True
):
    """
    Compute minWER loss .
    This implementation is based on the paper: https://arxiv.org/pdf/1712.01818.pdf (see section 3)

    Arguments
    ---------
    wers : torch.Tensor
        Tensor (B, N) of the number of word errors for each utterance
    hypotheses_scores : torch.Tensor
        Tensor (B, N) where N is the maximum
        length of hypotheses from batch.
    subtract_avg : bool
        Subtract the average number of word errors (a form of variance reduction)
    Returns
    -------
    torch.tensor
        minWER loss
    """
    wers = wers.to(hypotheses_scores.device)
    hypotheses_score_sums = torch.logsumexp(hypotheses_scores.detach(), dim=1, keepdim=True)
    hypotheses_scores = hypotheses_scores - hypotheses_score_sums
    if subtract_avg:
        avg_wers = torch.mean(wers, dim=1, keepdim=True)
        relative_wers = wers - avg_wers
        mWER_loss = torch.sum(hypotheses_scores.exp() * relative_wers, -1)
    else:
        mWER_loss = torch.sum(hypotheses_scores.exp() * wers, -1)

    return mWER_loss.mean()

