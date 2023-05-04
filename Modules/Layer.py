import torch
from typing import Optional

class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain: Optional[str]= None, *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain is None:
            super().reset_parameters()
        elif self.w_init_gain in ['zero']:
            torch.nn.init.zeros_(self.weight)
        elif self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        elif self.w_init_gain == 'gate':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.xavier_uniform_(self.weight[:self.out_channels // 2], gain= torch.nn.init.calculate_gain('tanh'))
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class ConvTranspose1d(torch.nn.ConvTranspose1d):
    def __init__(self, w_init_gain: Optional[str]= None, *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain is None:
            super().reset_parameters()
        elif self.w_init_gain in ['zero']:
            torch.nn.init.zeros_(self.weight)
        elif self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        elif self.w_init_gain == 'gate':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.xavier_uniform_(self.weight[:self.out_channels // 2], gain= torch.nn.init.calculate_gain('tanh'))
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Conv2d(torch.nn.Conv2d):
    def __init__(self, w_init_gain: Optional[str]= None, *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain is None:
            super().reset_parameters()
        elif self.w_init_gain in ['zero']:
            torch.nn.init.zeros_(self.weight)
        elif self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        elif self.w_init_gain == 'gate':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.xavier_uniform_(self.weight[:self.out_channels // 2], gain= torch.nn.init.calculate_gain('tanh'))
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self, w_init_gain: Optional[str]= None, *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain is None:
            super().reset_parameters()
        elif self.w_init_gain in ['zero']:
            torch.nn.init.zeros_(self.weight)
        elif self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        elif self.w_init_gain == 'gate':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.xavier_uniform_(self.weight[:self.out_channels // 2], gain= torch.nn.init.calculate_gain('tanh'))
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Linear(torch.nn.Linear):
    def __init__(self, w_init_gain: Optional[str]= None, *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain is None:
            super().reset_parameters()
        elif self.w_init_gain in ['zero']:
            torch.nn.init.zeros_(self.weight)
        elif self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_features % 2 == 0, 'The out_features of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_features // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_features // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Lambda(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class LayerNorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float= 1e-5):
        super().__init__()
        
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))


    def forward(self, inputs: torch.Tensor):
        means = inputs.mean(dim= 1, keepdim= True)
        variances = (inputs - means).pow(2.0).mean(dim= 1, keepdim= True)

        x = (inputs - means) * (variances + self.eps).rsqrt()

        shape = [1, -1] + [1] * (x.ndim - 2)

        return x * self.gamma.view(*shape) + self.beta.view(*shape)
      
class LightweightConv1d(torch.nn.Module):
    '''
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution

    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    '''

    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding=0,
        num_heads=1,
        weight_softmax=False,
        bias=False,
        weight_dropout=0.0,
        w_init_gain= 'linear'
    ):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = torch.nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        self.w_init_gain = w_init_gain

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.weight_dropout_module = FairseqDropout(
            weight_dropout, module_name=self.__class__.__name__
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        """
        input size: B x C x T
        output size: B x C x T
        """
        B, C, T = input.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = weight.softmax(dim=-1)

        weight = self.weight_dropout_module(weight)
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, H, T)
        output = torch.nn.functional.conv1d(input, weight, padding=self.padding, groups=self.num_heads)
        output = output.view(B, C, T)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output

class FairseqDropout(torch.nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.training or self.apply_during_inference:
            return torch.nn.functional.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]