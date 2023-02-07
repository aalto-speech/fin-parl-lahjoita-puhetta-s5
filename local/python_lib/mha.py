#!/usr/bin/env python3

from speechbrain.nnet.attention import LocationAwareAttention
import torch




"""
    def __init__(
        self,
        enc_dim,
        dec_dim,
        attn_dim,
        output_dim,
        conv_channels,
        kernel_size,
        scaling=1.0,
    ):
"""

class MultiHeadLocationAwareAttention(torch.nn.Module):

    def __init__(
        self,
        enc_dim,
        dec_dim,
        attn_dim,
        output_dim,
        conv_channels,
        kernel_size,
        scaling=1.0,
        num_heads=4
    ):
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("MHA output dim needs to divide by number of heads!")
        perhead_out_dim = output_dim // num_heads
        self.num_heads = num_heads
        self.attentions = torch.nn.ModuleList()
        for _ in range(num_heads):
            attention = LocationAwareAttention(
                    enc_dim,
                    dec_dim,
                    attn_dim,
                    perhead_out_dim,
                    conv_channels,
                    kernel_size,
                    scaling
                    )
            self.attentions.append(attention)
        self.reset()

    def reset(self):
        for attention in self.attentions:
            attention.reset()

    def forward(self, enc_states, enc_len, dec_states):
        context_vectors = []
        attn_weight = torch.zeros(enc_states.shape[0], enc_states.shape[1], device = enc_states.device)
        for attention in self.attentions:
            c, w = attention(enc_states, enc_len, dec_states)
            context_vectors.append(c)
            attn_weight += w
        context = torch.cat(context_vectors, dim=1)
        attn_weight /= self.num_heads
        return context, attn_weight
