import torch
import logging, sys
from typing import Optional

from .Layer import Conv1d, Mask_Generate

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class FlowBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        flow_stack: int,
        flow_wavenet_conv_stack: int,
        flow_wavenet_kernel_size: int,
        flow_wavnet_dilation_rate: int,
        flow_wavenet_dropout_rate: float= 0.0,
        condition_channels: Optional[int]= None,
        ):
        super().__init__()

        self.flows = torch.nn.ModuleList()
        for _ in range(flow_stack):
            self.flows.append(Flow(
                channels= channels,
                calc_channels= calc_channels,
                wavenet_conv_stack= flow_wavenet_conv_stack,
                wavenet_kernel_size= flow_wavenet_kernel_size,
                wavnet_dilation_rate= flow_wavnet_dilation_rate,
                wavenet_dropout_rate= flow_wavenet_dropout_rate,
                condition_channels= condition_channels,
                ))
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        conditions: Optional[torch.Tensor]= None,
        reverse: bool= False
        ):
        '''
        x: [Batch, Dim, Time]
        lengths: [Batch],
        conditions: [Batch, Cond_d], This may be a speaker or emotion embedding vector.
        reverse: a boolean
        '''
        masks= (~Mask_Generate(lengths, max_length= x.size(2))).unsqueeze(1).float()
        if not conditions is None:
            conditions = conditions.unsqueeze(2)    # [Batch, Cond_d, 1]
        
        for flow in (self.flows if not reverse else reversed(self.flows)):
            x = flow(
                x= x,
                masks= masks,
                conditions= conditions,
                reverse= reverse,
                )

        return x

class Flow(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        wavenet_conv_stack: int,
        wavenet_kernel_size: int,
        wavnet_dilation_rate: int,
        wavenet_dropout_rate: float= 0.0,
        condition_channels: Optional[int]= None,
        ):
        super().__init__()
        assert channels % 2 == 0, 'in_channels must be a even.'

        self.prenet = Conv1d(
            in_channels= channels // 2,
            out_channels= calc_channels,
            kernel_size= 1,
            # w_init_gain= 'linear' # Don't use init. This become a reason lower quality.
            )
        
        self.wavenet = WaveNet(
            calc_channels= calc_channels,
            conv_stack= wavenet_conv_stack,
            kernel_size= wavenet_kernel_size,
            dilation_rate= wavnet_dilation_rate,
            dropout_rate= wavenet_dropout_rate,
            condition_channels= condition_channels,
            )
        
        self.postnet = Conv1d(
            in_channels= calc_channels,
            out_channels= channels // 2,
            kernel_size= 1,
            w_init_gain= 'zero'
            )

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        conditions: Optional[torch.Tensor]= None,
        reverse: bool= False
        ):
        x_0, x_1 = x.chunk(chunks= 2, dim= 1)   # [Batch, Dim // 2, Time] * 2
        x_hiddens = self.prenet(x_0) * masks    # [Batch, Calc_d, Time]
        x_hiddens = self.wavenet(
            x= x_hiddens,
            masks= masks,
            conditions= conditions
            )   # [Batch, Calc_d, Time]
        means = self.postnet(x_hiddens) # [Batch, Dim // 2, Time]

        if not reverse:
            x_1 = (x_1 + means) * masks
        else:
            x_1 = (x_1 - means) * masks

        x = torch.cat([x_0, x_1], dim= 1)   # [Batch, Dim, Time]

        return x

class Flip(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
        ):
        return x.flip(dims= [1,])

class WaveNet(torch.nn.Module):
    def __init__(
        self,
        calc_channels: int,
        conv_stack: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float= 0.0,
        condition_channels: Optional[int]= None,
        ):
        super().__init__()
        self.calc_channels = calc_channels
        self.conv_stack = conv_stack
        self.use_condition = not condition_channels is None

        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)

        if self.use_condition:
            self.condition = torch.nn.utils.weight_norm(Conv1d(
                in_channels= condition_channels,
                out_channels= calc_channels * conv_stack * 2,
                kernel_size= 1
                ))
            self.condition.apply(weight_norm_initialize_weight)
        
        self.input_convs = torch.nn.ModuleList()
        self.residual_and_skip_convs = torch.nn.ModuleList()
        for index in range(conv_stack):
            dilation = dilation_rate ** index
            padding = (kernel_size - 1) * dilation // 2
            self.input_convs.append(torch.nn.utils.weight_norm(Conv1d(
                in_channels= calc_channels,
                out_channels= calc_channels * 2,
                kernel_size= kernel_size,
                dilation= dilation,
                padding= padding
                )))
            self.residual_and_skip_convs.append(torch.nn.utils.weight_norm(Conv1d(
                in_channels= calc_channels,
                out_channels= calc_channels * 2,
                kernel_size= 1
                )))

        self.dropout = torch.nn.Dropout(p= dropout_rate)
        
        self.input_convs.apply(weight_norm_initialize_weight)
        self.residual_and_skip_convs.apply(weight_norm_initialize_weight)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        conditions: Optional[torch.Tensor]= None,    
        ):
        if self.use_condition:
            conditions_list = self.condition(conditions).chunk(chunks= self.conv_stack, dim= 1)  # [Batch, Calc_d * 2, Time] * Stack
        else:
            conditions_list = [torch.zeros(
                size= (x.size(0), self.calc_channels * 2, x.size(2)),
                dtype= x.dtype,
                device= x.device
                )] * self.conv_stack

        skips_list = []
        for in_conv, conditions, residual_and_skip_conv in zip(self.input_convs, conditions_list, self.residual_and_skip_convs):
            ins = in_conv(x)
            acts = Fused_Gate(ins + conditions)
            acts = self.dropout(acts)
            residuals, skips = residual_and_skip_conv(acts).chunk(chunks= 2, dim= 1)
            x = (x + residuals) * masks
            skips_list.append(skips)

        skips = torch.stack(skips_list, dim= 1).sum(dim= 1) * masks

        return skips

@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x

def Flow_KL_Loss(encoding_means, encoding_log_stds, flows, flow_log_stds, masks):
    encoding_means = encoding_means.float()
    encoding_log_stds = encoding_log_stds.float()
    flows = flows.float()
    flow_log_stds = flow_log_stds.float()
    masks = masks.float()

    loss = encoding_log_stds - flow_log_stds - 0.5
    loss += 0.5 * (flows - encoding_means).pow(2.0) * (-2.0 * encoding_log_stds).exp()
    loss = (loss * masks).sum() / masks.sum()
    
    return loss