from argparse import Namespace
import torch
import numpy as np
import math
from numba import jit
from typing import Optional, List, Dict, Tuple, Union

from .LinearAttention import LinearAttention
from .Layer import Conv1d, Linear, Lambda, Residual, LayerNorm

class Monotonic_Alignment_Search(torch.nn.Module): 
    def __init__(
        self,
        in_channels: int,        
        feature_size: int,
        condition_channels: int,
        condition_attenion_head: int
        ):
        super().__init__()

        self.attention = LinearAttention(
            query_channels= in_channels,
            key_channels= condition_channels, 
            value_channels= condition_channels,
            calc_channels= in_channels,
            num_heads= condition_attenion_head
            )

        self.vae_projection = Conv1d(
            in_channels= in_channels,
            out_channels= feature_size * 2,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
        self.maximum_path_generator = Maximum_Path_Generator()

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor,
        conditions: torch.Tensor,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        encoding_lengths: [Batch]
        features: [Batch, Feature_d, Feature_t]
        feature_lengths: [Batch]
        '''
        encoding_masks = (~Mask_Generate(lengths= encoding_lengths, max_length= torch.ones_like(encodings[0, 0]).sum())).unsqueeze(1).float()
        feature_masks = (~Mask_Generate(lengths= feature_lengths, max_length= torch.ones_like(features[0, 0]).sum())).unsqueeze(1).float()

        encodings = self.attention(
            queries= encodings,
            keys= conditions,
            values= conditions
            )

        means, stds = (self.vae_projection(encodings * encoding_masks) * encoding_masks).chunk(chunks= 2, dim= 1)
        log_stds = torch.nn.functional.softplus(stds).log()

        with torch.no_grad():
            # negative cross-entropy
            stds_sq_r = torch.exp(-2 * log_stds) # [Batch, Enc_d, Token_t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - log_stds, [1], keepdim=True) # [Batch, 1, Token_t]
            neg_cent2 = torch.matmul(-0.5 * (features ** 2).permute(0, 2, 1), stds_sq_r) # [Batch, Feature_t, Enc_d] x [Batch, Enc_d, Token_t] -> [Batch, Feature_t, Token_t]
            neg_cent3 = torch.matmul(features.permute(0, 2, 1), (means * stds_sq_r)) # [Batch, Feature_t, Enc_d] x [b, Enc_d, Token_t] -> [Batch, Feature_t, Token_t]
            neg_cent4 = torch.sum(-0.5 * (means ** 2) * stds_sq_r, [1], keepdim=True) # [Batch, 1, Token_t]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4    # [Batch, Feature_t, Token_t]

            attention_masks = encoding_masks * feature_masks.permute(0, 2, 1)  # [Batch, 1, Token_t] x [Batch, Feature_t, 1] -> [Batch, Feature_t, Token_t]
            alignments = self.maximum_path_generator(neg_cent, attention_masks).detach()    # [Batch, Feature_t, Token_t]

        durations = self.Calc_Duration_from_Alignment(
            alignments= alignments,
            encoding_lengths= encoding_lengths,
            feature_lengths= feature_lengths
            )
        alignments = alignments.permute(0, 2, 1)

        return alignments, durations, means, log_stds

    def Calc_Duration_from_Alignment(self, alignments, encoding_lengths, feature_lengths):
        '''
        alignemnts: [Batch, Feature_t, Enc_t]
        encoding_lengths: [Batch]
        feature_lengths: [Batch]
        '''
        durations = [
            torch.nn.functional.one_hot(
                alignment[:feature_length, :encoding_length].argmax(dim= 1),
                num_classes= alignments.size(2)
                ).sum(dim= 0)
            for alignment, encoding_length, feature_length in zip(alignments, encoding_lengths, feature_lengths)
            ]
        for duration, encoding_length in zip(durations, encoding_lengths):
            duration[encoding_length - 1] += alignments.size(1) - duration.sum()
        durations = torch.stack(durations, dim= 0)

        return durations



class Maximum_Path_Generator(torch.nn.Module):
    def forward(self, neg_cent, mask):
        '''
        x: [Batch, Feature_t, Token_t]
        mask: [Batch, Feature_t, Token_t]
        '''
        neg_cent *= mask
        device, dtype = neg_cent.device, neg_cent.dtype
        neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
        mask = mask.data.cpu().numpy()

        token_lengths = mask.sum(axis= 2)[:, 0].astype('int32')   # [Batch]
        feature_lengths = mask.sum(axis= 1)[:, 0].astype('int32')   # [Batch]

        paths = self.calc_paths(neg_cent, token_lengths, feature_lengths)

        return torch.from_numpy(paths).to(device= device, dtype= dtype)

    def calc_paths(self, neg_cent, token_lengths, feature_lengths):
        return np.stack([
            Maximum_Path_Generator.calc_path(x, token_length, feature_length)
            for x, token_length, feature_length in zip(neg_cent, token_lengths, feature_lengths)
            ], axis= 0)

    @staticmethod
    @jit(nopython=True)
    def calc_path(x, token_length, feature_length):
        path = np.zeros_like(x, dtype= np.int32)
        for feature_index in range(feature_length):
            for token_index in range(max(0, token_length + feature_index - feature_length), min(token_length, feature_index + 1)):
                if feature_index == token_index:
                    current_q = -1e+9
                else:
                    current_q = x[feature_index - 1, token_index]   # Stayed current token
                if token_index == 0:
                    if feature_index == 0:
                        prev_q = 0.0
                    else:
                        prev_q = -1e+9
                else:
                    prev_q = x[feature_index - 1, token_index - 1]  # Moved to next token
                x[feature_index, token_index] = x[feature_index, token_index] + max(prev_q, current_q)

        token_index = token_length - 1
        for feature_index in range(feature_length - 1, -1, -1):
            path[feature_index, token_index] = 1
            if token_index != 0 and token_index == feature_index or x[feature_index - 1, token_index] < x[feature_index - 1, token_index - 1]:
                token_index = token_index - 1

        return path

def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]

def MAS_MLE_Loss(
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
    means: torch.Tensor,
    log_stds: torch.Tensor
    ):
    feature_masks = (~Mask_Generate(
        lengths= feature_lengths,
        max_length= torch.ones_like(features[0, 0]).sum()
        )).unsqueeze(1).float()

    features = features * feature_masks
    means = means * feature_masks
    log_stds = log_stds * feature_masks

    loss = torch.sum(log_stds) + 0.5 * torch.sum(torch.exp(-2 * log_stds) * ((features - means)**2)) # neg normal likelihood w/o the constant term
    loss = loss / torch.sum(torch.ones_like(features) * feature_masks) # averaging across batch, channel and time axes
    loss = loss + 0.5 * math.log(2 * math.pi) # add the remaining constant term

    return loss
