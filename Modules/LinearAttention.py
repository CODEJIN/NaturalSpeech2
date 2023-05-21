import torch
from typing import Optional
from einops import rearrange, einsum

from .Layer import Conv1d, RMSNorm, Lambda

class LinearAttention(torch.nn.Module):
    def __init__(
        self,
        query_channels: int,
        key_channels: int,
        value_channels: int,
        calc_channels: int,
        num_heads: int,       
        ):
        super().__init__()
        assert calc_channels % num_heads == 0
        self.calc_channels = calc_channels
        self.num_heads = num_heads

        self.query_scale = num_heads ** -0.5
        
        self.query = torch.nn.Sequential(
            Conv1d(
                in_channels= query_channels,
                out_channels= calc_channels,
                kernel_size= 1,
                w_init_gain= 'linear'
                ),
            RMSNorm(num_features= calc_channels),
            Lambda(lambda x: torch.nn.functional.elu(x) + 1.0)
            )
        self.key = torch.nn.Sequential(
            Conv1d(
                in_channels= key_channels,
                out_channels= calc_channels,
                kernel_size= 1,
                w_init_gain= 'linear'
                ),
            RMSNorm(num_features= calc_channels),
            Lambda(lambda x: torch.nn.functional.elu(x) + 1.0)
            )
        self.value = torch.nn.Sequential(
            Conv1d(
                in_channels= value_channels,
                out_channels= calc_channels,
                kernel_size= 1,
                w_init_gain= 'linear'
                ),
            RMSNorm(num_features= calc_channels),
            Lambda(lambda x: torch.nn.functional.elu(x) + 1.0)
            )
        
        self.projection = Conv1d(
            in_channels= calc_channels,
            out_channels= query_channels,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
        
        self.norm = RMSNorm(num_features= query_channels)


    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_padding_masks: Optional[torch.Tensor]= None,
        *args,
        **kwargs
        ):
        '''
        queries: [Batch, Enc_d, Enc_t]
        keys: [Batch, Enc_d, Key_t]
        values: [Batch, Enc_d, Key_t]
        key_padding_masks: [Batch, Key_t]
        '''
        residuals = queries

        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        queries = rearrange(queries, 'batch (head dimension) time -> batch head dimension time', head= self.num_heads)
        keys = rearrange(keys, 'batch (head dimension) time -> batch head dimension time', head= self.num_heads)
        values = rearrange(values, 'batch (head dimension) time -> batch head dimension time', head= self.num_heads)
        
        if not key_padding_masks is None:
            keys.masked_fill_(key_padding_masks[:, None, None, :], -1e+4)

        queries = queries.softmax(dim= 2) * self.query_scale    # channel softmax
        keys = keys.softmax(dim= 3)    # time softmax
        values = values / values.size(2)
        
        contexts = einsum(keys, values, 'batch head key_d time, batch head value_d time -> batch head key_d value_d')
        contexts = einsum(queries, contexts, 'batch head query_d time, batch head query_d value_d -> batch head value_d time')
        contexts = rearrange(contexts, 'batch head dimension time -> batch (head dimension) time')
        contexts = self.projection(contexts)    # [Batch, Enc_d, Enc_t]
        contexts = self.norm(contexts + residuals)

        return contexts