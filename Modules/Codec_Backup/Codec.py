from argparse import Namespace
import torch
from typing import List
from einops import rearrange

from .Layer import Conv1d, ConvTranspose1d

class Encoder(torch.nn.Sequential):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.append(Conv1d(
            in_channels= 1,
            out_channels= 1,
            kernel_size= self.hp.Audio_Codec.Encoder.Kernel_Size.Initial,
            padding= (self.hp.Audio_Codec.Encoder.Kernel_Size.Initial - 1) // 2,
            w_init_gain= 'linear'
            ))
        self.append(torch.nn.ELU())

        previous_channels = 1
        for channels, stride in zip(
            self.hp.Audio_Codec.Encoder.Channels,
            self.hp.Audio_Codec.Encoder.Strides
            ):
            self.append(Encoder_Block(
                in_channels= previous_channels,
                out_channels= channels,
                stride= stride
                ))
            previous_channels = channels
            
        self.append(Conv1d(
            in_channels= previous_channels,
            out_channels= self.hp.Audio_Codec.Size,
            kernel_size= self.hp.Audio_Codec.Encoder.Kernel_Size.Last,
            padding= (self.hp.Audio_Codec.Encoder.Kernel_Size.Last - 1) // 2,
            w_init_gain= 'linear'
            ))

class Encoder_Block(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        residual_block_kernel_size: int= 7,
        residual_block_dilation: List[int]= [1, 3, 9]
        ):
        super().__init__()
        
        previous_channels = in_channels
        for dilation in residual_block_dilation:
            self.append(Residual_Block(
                in_channels= previous_channels,
                out_channels= out_channels // 2,
                kernel_size= residual_block_kernel_size,
                dilation= dilation,
                ))
            self.append(torch.nn.ELU())
            previous_channels = out_channels // 2

        self.append(Conv1d(
            in_channels= previous_channels,
            out_channels= out_channels,
            kernel_size= stride * 2 - 1,
            stride= stride,
            padding= (stride * 2 - 1 - 1) // 2
            ))
        self.append(torch.nn.ELU())


class Decoder(torch.nn.Sequential):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.append(Conv1d(
            in_channels= self.hp.Audio_Codec.Size,
            out_channels=self.hp.Audio_Codec.Size,
            kernel_size= self.hp.Audio_Codec.Decoder.Kernel_Size.Initial,
            padding= (self.hp.Audio_Codec.Decoder.Kernel_Size.Initial - 1) // 2,
            w_init_gain= 'linear'
            ))
        self.append(torch.nn.ELU())

        previous_channels = self.hp.Audio_Codec.Size
        for channels, stride in zip(
            self.hp.Audio_Codec.Decoder.Channels,
            self.hp.Audio_Codec.Decoder.Strides
            ):
            self.append(Decoder_Block(
                in_channels= previous_channels,
                out_channels= channels,
                stride= stride
                ))
            previous_channels = channels
            
        self.append(Conv1d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= self.hp.Audio_Codec.Decoder.Kernel_Size.Last,
            padding= (self.hp.Audio_Codec.Decoder.Kernel_Size.Last - 1) // 2,
            w_init_gain= 'linear'
            ))

class Decoder_Block(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        residual_block_kernel_size: int= 7,
        residual_block_dilation: List[int]= [1, 3, 9]
        ):
        super().__init__()
        self.stride = stride

        self.append(ConvTranspose1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= stride * 2,
            stride= stride,
            padding= (stride * 2 - stride) // 2,
            w_init_gain= 'linear'
            ))
        self.append(torch.nn.ELU())
        
        for dilation in residual_block_dilation:
            self.append(Residual_Block(
                in_channels= out_channels,
                out_channels= out_channels,
                kernel_size= residual_block_kernel_size,
                dilation= dilation,
                ))
            self.append(torch.nn.ELU())

    def forward(self, x: torch.Tensor):
        output_size = x.size(2) * self.stride
        return super().forward(x)[:, :, :output_size]


class Residual_Block(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int
        ):
        super().__init__()
        self.append(Conv1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            dilation= dilation,
            padding= (kernel_size - 1) * dilation // 2,
            w_init_gain= 'linear'
            ))
        self.append(torch.nn.ELU())
        self.append(Conv1d(
            in_channels= out_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            ))
        
    def forward(self, x: torch.Tensor):
        return x + super().forward(x)
