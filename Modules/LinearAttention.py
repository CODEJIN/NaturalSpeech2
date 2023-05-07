import torch
from typing import Optional

from .Layer import Conv1d, LayerNorm

class LinearAttention(torch.nn.Module):
    def __init__(
        self,
        query_channels: int,
        key_channels: int,
        value_channels: int,
        calc_channels: int,
        num_heads: int,
        dropout_rate: float= 0.0,
        use_scale: bool= True,
        use_residual: bool= True,
        use_norm: bool= True
        ):
        super().__init__()
        assert calc_channels % num_heads == 0
        self.calc_channels = calc_channels
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.use_residual = use_residual
        self.use_norm = use_norm

        self.query = Conv1d(
            in_channels= query_channels,
            out_channels= calc_channels,
            kernel_size= 1,
            bias=False,
            w_init_gain= 'linear'
            )
        self.key = Conv1d(
            in_channels= key_channels,
            out_channels= calc_channels,
            kernel_size= 1,
            bias=False,
            w_init_gain= 'linear'
            )
        self.value = Conv1d(
            in_channels= value_channels,
            out_channels= calc_channels,
            kernel_size= 1,
            bias=False,
            w_init_gain= 'linear'
            )
        self.projection = Conv1d(
            in_channels= calc_channels,
            out_channels= query_channels,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        
        if use_scale:
            self.scale = torch.nn.Parameter(torch.zeros(1))

        if use_norm:
            self.norm = LayerNorm(num_features= query_channels)

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

        queries = queries.view(queries.size(0), self.num_heads, queries.size(1) // self.num_heads, queries.size(2))    # [Batch, Head, Calc_d // Head, Enc_t]
        keys = keys.view(keys.size(0), self.num_heads, keys.size(1) // self.num_heads, keys.size(2))    # [Batch, Head, Calc_d // Head, Enc_t]
        values = values.view(values.size(0), self.num_heads, values.size(1) // self.num_heads, values.size(2))    # [Batch, Head, Calc_d // Head, Enc_t]
        
        if not key_padding_masks is None:
            keys.masked_fill_(key_padding_masks[:, None, None, :], -1e+4)

        keys = (keys + 1e-4).softmax(dim= 3)

        contexts = keys @ values.permute(0, 1, 3, 2)   # [Batch, Head, Calc_d // Head, Calc_d // Head]
        contexts = contexts.permute(0, 1, 3, 2) @ queries   # [Batch, Head, Calc_d // Head, Enc_t]
        contexts = contexts.view(contexts.size(0), contexts.size(1) * contexts.size(2), contexts.size(3))   # [Batch, Calc_d, Enc_t]
        contexts = self.projection(contexts)    # [Batch, Enc_d, Enc_t]

        if self.use_scale:
            contexts = self.scale * contexts

        contexts = self.dropout(contexts)

        if self.use_residual:
            contexts = contexts + residuals

        if self.use_norm:
            contexts = self.norm(contexts)

        return contexts
