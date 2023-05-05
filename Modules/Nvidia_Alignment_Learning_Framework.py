# BSD 3-Clause License
#
# Copyright (c) 2020, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from numba import jit
from typing import Optional

import functools
from scipy import ndimage
from scipy.stats import betabinom

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Invertible1x1ConvLUS(torch.nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvLUS, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        W, _ = torch.linalg.qr(torch.randn(c, c))
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.lu(W))

        self.register_buffer('p', p)
        # diagonals of lower will always be 1s anyway
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c, c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))

    def forward(self, z, reverse=False):
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z, log_det_W

class ConvAttention(torch.nn.Module):
    def __init__(self, n_mel_channels=80, n_speaker_dim=128,
                 n_text_channels=512, n_att_channels=80, temperature=1.0,
                 n_mel_convs=2, align_query_enc_type='3xconv',
                 use_query_proj=True):
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        self.att_scaling_factor = np.sqrt(n_att_channels)
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        self.attn_proj = torch.nn.Conv2d(n_att_channels, 1, kernel_size=1)
        self.align_query_enc_type = align_query_enc_type
        self.use_query_proj = bool(use_query_proj)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels,
                     n_text_channels * 2,
                     kernel_size=3,
                     bias=True,
                     w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2,
                     n_att_channels,
                     kernel_size=1,
                     bias=True))

        self.align_query_enc_type = align_query_enc_type

        if align_query_enc_type == "inv_conv":
            self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        elif align_query_enc_type == "3xconv":
            self.query_proj = nn.Sequential(
                ConvNorm(n_mel_channels,
                         n_mel_channels * 2,
                         kernel_size=3,
                         bias=True,
                         w_init_gain='relu'),
                torch.nn.ReLU(),
                ConvNorm(n_mel_channels * 2,
                         n_mel_channels,
                         kernel_size=1,
                         bias=True),
                torch.nn.ReLU(),
                ConvNorm(n_mel_channels,
                         n_att_channels,
                         kernel_size=1,
                         bias=True))
        else:
            raise ValueError("Unknown query encoder type specified")

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model to run data through
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original,
            unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def encode_query(self, query, query_lens):
        query = query.permute(2, 0, 1)  # seq_len, batch, feature dim
        lens, ids = torch.sort(query_lens, descending=True)
        original_ids = [0] * lens.size(0)
        for i in range(len(ids)):
            original_ids[ids[i]] = i

        query_encoded = self.run_padded_sequence(ids, original_ids, lens,
                                                 query, self.query_lstm)
        query_encoded = query_encoded.permute(1, 2, 0)
        return query_encoded

    def forward(self, queries, keys, query_lens, mask=None, key_lens=None,
                keys_encoded=None, attn_prior=None):
        """Attention mechanism for flowtron parallel
        Unlike in Flowtron, we have no restrictions such as causality etc,
        since we only need this during training.

        Args:
            queries (torch.tensor): B x C x T1 tensor
                (probably going to be mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries
                (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                Final dim T2 should sum to 1
        """
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2

        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        if self.use_query_proj:
            if self.align_query_enc_type == "inv_conv":
                queries_enc, log_det_W = self.query_proj(queries)
            elif self.align_query_enc_type == "3xconv":
                queries_enc = self.query_proj(queries)
                log_det_W = 0.0
            else:
                queries_enc, log_det_W = self.query_proj(queries)
        else:
            queries_enc, log_det_W = queries, 0.0

        # different ways of computing attn,
        # one is isotopic gaussians (per phoneme)
        # Simplistic Gaussian Isotopic Attention

        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2
        # compute log likelihood from a gaussian
        attn = -0.0005 * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None]+1e-8)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2),
                                   -float("inf"))

        attn = self.softmax(attn)  # Softmax along T2
        return attn, attn_logprob


@jit(nopython=True)
def mas_width1(log_attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i-1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2

    # now backtrack
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1]-1
    for i in range(log_p.shape[0]-1, 0, -1):
        opt[i, j] = one
        if log_p[i-1, j-1] >= log_p[i-1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break
    opt[0, j] = one
    return opt

def binarize_attention(attn, in_lens, out_lens):
    """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.

    Args:
        attn: B x 1 x max_mel_len x max_text_len
    """
    b_size = attn.shape[0]
    with torch.no_grad():
        attn_out_cpu = np.zeros(attn.data.shape, dtype=np.float32)
        log_attn_cpu = torch.log(attn.data).to(device='cpu', dtype=torch.float32)
        log_attn_cpu = log_attn_cpu.numpy()
        out_lens_cpu = out_lens.cpu()
        in_lens_cpu = in_lens.cpu()
        for ind in range(b_size):
            hard_attn = mas_width1(
                log_attn_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]])
            attn_out_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]] = hard_attn
        attn_out = torch.tensor(
            attn_out_cpu, device=attn.get_device(), dtype=attn.dtype)
    return attn_out




class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret

class Attention_Prior_Generator:
    def __init__(self, use_betabinomial_interpolator: bool= True):
        self.use_betabinomial_interpolator = use_betabinomial_interpolator
        if use_betabinomial_interpolator:
            self.betabinomial_interpolator = BetaBinomialInterpolator()

    def get_prior(self, mel_len, text_len):
        if self.use_betabinomial_interpolator:
            return torch.from_numpy(self.betabinomial_interpolator(mel_len, text_len))

        attn_prior = beta_binomial_prior_distribution(text_len, mel_len)
        
        return attn_prior.numpy()

def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))

def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask




class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_logprob.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_logprob = attn_logprob.squeeze(1)
        attn_logprob = attn_logprob.permute(1, 0, 2)

        # Add blank label
        attn_logprob = F.pad(
            input=attn_logprob,
            pad=(1, 0, 0, 0, 0, 0),
            value=self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(
            max_key_len+1,
            device=attn_logprob.device,
            dtype=torch.long)
        attn_logprob.masked_fill_(
            key_inds.view(1,1,-1) > key_lens.view(1,-1,1), # key_inds >= key_lens+1
            -float("inf"))
        attn_logprob = self.log_softmax(attn_logprob)

        # Target sequences
        target_seqs = key_inds[1:].unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.CTCLoss(
            attn_logprob, target_seqs,
            input_lengths=query_lens, target_lengths=key_lens)
        return cost

class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention, eps=1e-12):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1],
                            min=eps)).sum()
        return -log_sum / hard_attention.sum()



class Alignment_Learning_Framework(torch.nn.Module):
    def __init__(
        self,
        feature_size: int,
        encoding_size: int
        ):
        super().__init__()

        self.attention = ConvAttention(
            feature_size,
            0,
            encoding_size,
            use_query_proj=True,
            align_query_enc_type='3xconv'
            )
        
    def forward(
        self,
        token_embeddings: torch.Tensor,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        attention_priors: torch.Tensor
        ):
        attention_masks = mask_from_lens(encoding_lengths, max_len=encoding_lengths.max())
        attention_masks = attention_masks[..., None] == 0
        
        attention_softs, attention_logprobs = self.attention(
            queries= features,
            keys= token_embeddings,
            query_lens= feature_lengths,
            mask= attention_masks,
            key_lens=encoding_lengths,
            keys_encoded= encodings,
            attn_prior= attention_priors
            )
        attention_hards = binarize_attention(attention_softs, encoding_lengths, feature_lengths)

        durations = attention_hards.sum(2)[:, 0, :]
        assert torch.all(torch.eq(durations.sum(dim=1), feature_lengths))

        return durations, attention_softs, attention_hards, attention_logprobs