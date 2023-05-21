import torch
import numpy as np
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm

from .LinearAttention import LinearAttention
from .Layer import Conv1d, Lambda

class Diffusion(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.denoiser = Denoiser(
            hyper_parameters= self.hp
            )

        self.gamma_scheuler = sigmoid_schedule

    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.FloatTensor,
        latents: Optional[torch.Tensor]= None
        ):
        '''
        encodings: [Batch, Enc_d, Audio_ct]
        latents: [Batch, Latent_d, Audio_ct]
        '''
        if not latents is None:    # train
            diffusion_targets, diffusion_predictions, diffusion_starts = self.Train(
                latents= latents,
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts
                )
            return None, diffusion_targets, diffusion_predictions, diffusion_starts
        else:   # inference
            latents = self.DDPM(
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts,
                )
            return latents, None, None, None

    def Train(
        self,
        latents: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.FloatTensor,
        ):
        noises = torch.randn_like(latents)

        diffusion_steps = torch.rand(
            size= (latents.size(0), ),
            device= latents.device
            )
        gammas = self.gamma_scheuler(diffusion_steps)
        alphas, sigmas = self.gamma_to_alpha_sigma(gammas)

        noised_latents = latents * alphas[:, None, None] + noises * sigmas[:, None, None]
        diffusion_predictions = self.denoiser(
            latents= noised_latents,
            encodings= encodings,
            lengths= lengths,
            speech_prompts= speech_prompts,
            diffusion_steps= diffusion_steps,
            )
        
        diffusion_targets = noises * alphas[:, None, None] - latents * sigmas[:, None, None]
        diffusion_starts = latents * alphas[:, None, None] - diffusion_predictions * sigmas[:, None, None]
        
        return diffusion_targets, diffusion_predictions, diffusion_starts

    def DDPM(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.Tensor,
        eps: float= 1e-7    # minimum at float16 precision
        ):
        steps = self.Get_Sampling_Steps(
            steps= self.hp.Diffusion.Max_Step,
            references= encodings
            )

        latents = torch.randn(
            size= (encodings.size(0), self.hp.Audio_Codec.Size, encodings.size(2)),
            device= encodings.device
            )

        for current_steps, next_steps in steps:            
            gammas = self.gamma_scheuler(current_steps)
            alphas, sigmas = self.gamma_to_alpha_sigma(gammas)
            log_snrs = self.gamma_to_log_snr(gammas)
            alphas, sigmas, log_snrs = alphas[:, None, None], sigmas[:, None, None], log_snrs[:, None, None]

            next_gammas = self.gamma_scheuler(next_steps)
            next_alphas, next_sigmas = self.gamma_to_alpha_sigma(next_gammas)
            next_log_snrs = self.gamma_to_log_snr(next_gammas)
            next_alphas, next_sigmas, next_log_snrs = next_alphas[:, None, None], next_sigmas[:, None, None], next_log_snrs[:, None, None]

            coefficients = -torch.expm1(log_snrs - next_log_snrs)
            
            noised_predictions = self.denoiser(
                latents= latents,
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts,
                diffusion_steps= current_steps,
                )

            epsilons = latents * alphas - noised_predictions * sigmas
            # epsilons.clamp_(-1.0, 1.0)  # clipped

            posterior_means = next_alphas * (latents * (1.0 - coefficients) / alphas + coefficients * epsilons)
            posterior_log_varainces = torch.log(torch.clamp(next_sigmas ** 2 * coefficients, min= eps))

            noises = torch.randn_like(latents)
            masks = (current_steps > 0).float().unsqueeze(1).unsqueeze(1) #[Batch, 1, 1]
            latents = posterior_means + masks * (0.5 * posterior_log_varainces).exp() * noises

        return latents
            
    def DDIM(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.Tensor,
        ddim_steps: int
        ):
        steps = self.Get_Sampling_Steps(
            steps= ddim_steps,
            references= encodings
            )

        latents = torch.randn(
            size= (encodings.size(0), self.hp.Audio_Codec.Size, encodings.size(2)),
            device= encodings.device
            )

        for current_steps, next_steps in steps:            
            gammas = self.gamma_scheuler(current_steps)
            alphas, sigmas = self.gamma_to_alpha_sigma(gammas)
            alphas, sigmas = alphas[:, None, None], sigmas[:, None, None]

            next_gammas = self.gamma_scheuler(next_steps)
            next_alphas, next_sigmas = self.gamma_to_alpha_sigma(next_gammas)
            next_alphas, next_sigmas = next_alphas[:, None, None], next_sigmas[:, None, None]
            
            noised_predictions = self.denoiser(
                latents= latents,
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts,
                diffusion_steps= current_steps,
                )
            epsilons = latents * alphas - noised_predictions * sigmas
            # epsilons.clamp_(-1.0, 1.0)  # clipped

            noises = (latents - alphas * epsilons) / sigmas            
            latents = epsilons * next_alphas + noises * next_sigmas

        return latents

    def Get_Sampling_Steps(
        self,        
        steps: int,
        references: torch.Tensor
        ):
        steps = torch.linspace(
            start= 1.0,
            end= 0.0,
            steps= steps + 1,
            device= references.device
            )   # [Step + 1]
        steps = torch.stack([steps[:-1], steps[1:]], dim= 0) # [2, Step]
        steps = steps.unsqueeze(1).expand(-1, references.size(0), -1)    # [2, Batch, Step]
        steps = steps.unbind(dim= 2)

        return steps
    
    def gamma_to_alpha_sigma(self, gamma, scale = 1):
        return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

    def gamma_to_log_snr(self, gamma, scale = 1, eps = 1e-7):
        return torch.log(torch.clamp(gamma * (scale ** 2) / (1 - gamma), min= eps))

class Denoiser(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Audio_Codec.Size,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.SiLU()
            )

        self.encoding_ffn = torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.Encoder.Size * 4,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.SiLU(),
            Conv1d(
                in_channels= self.hp.Encoder.Size * 4,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                w_init_gain= 'linear'
                )
            )
        self.step_ffn = torch.nn.Sequential(
            Diffusion_Embedding(
                channels= self.hp.Diffusion.Size
                ),
            Lambda(lambda x: x.unsqueeze(2)),
            Conv1d(
                in_channels= self.hp.Diffusion.Size + 1,
                out_channels= self.hp.Diffusion.Size * 4,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.SiLU(),
            Conv1d(
                in_channels= self.hp.Diffusion.Size * 4,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                w_init_gain= 'linear'
                )
            )
        
        self.pre_attention = LinearAttention(
            query_channels= self.hp.Diffusion.Pre_Attention.Query_Size,
            key_channels= self.hp.Speech_Prompter.Size, 
            value_channels= self.hp.Speech_Prompter.Size,
            calc_channels= self.hp.Diffusion.Pre_Attention.Query_Size,
            num_heads= self.hp.Diffusion.Pre_Attention.Head
            )

        self.pre_attention_query = torch.nn.Parameter(
            torch.empty(1, self.hp.Diffusion.Pre_Attention.Query_Size, self.hp.Diffusion.Pre_Attention.Query_Token)
            )
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (self.hp.Diffusion.Pre_Attention.Query_Size + self.hp.Diffusion.Pre_Attention.Query_Token))
        self.pre_attention_query.data.uniform_(-query_variance, query_variance)
        
        
        self.wavenets = torch.nn.ModuleList([
            WaveNet(
                channels= self.hp.Diffusion.Size,
                kernel_size= self.hp.Diffusion.WaveNet.Kernel_Size,
                dilation= self.hp.Diffusion.WaveNet.Dilation,
                condition_channels= self.hp.Diffusion.Size,
                diffusion_step_channels= self.hp.Diffusion.Size,
                wavenet_dropout_rate= self.hp.Diffusion.WaveNet.Dropout_Rate,
                apply_film= (wavenet_index + 1) % self.hp.Diffusion.WaveNet.Attention.Apply_in_Stack == 0,
                speech_prompt_channels= self.hp.Speech_Prompter.Size,
                speech_prompt_attention_head= self.hp.Diffusion.WaveNet.Attention.Head
                )
            for wavenet_index in range(self.hp.Diffusion.WaveNet.Stack)
            ])

        self.postnet = torch.nn.Sequential(
            torch.nn.SiLU(),
            Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Audio_Codec.Size,
                kernel_size= 1,
                w_init_gain= 'zero'
                )
            ) 

    def forward(
        self,
        latents: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.Tensor,
        diffusion_steps: torch.Tensor,
        ):
        '''
        latents: [Batch, Codec_d, Audio_ct]
        encodings: [Batch, Enc_d, Audio_ct]
        diffusion_steps: [Batch]
        speech_prompts: [Batch, Prompt_d, Prompt_t]
        '''
        masks= (~Mask_Generate(lengths, max_length= latents.size(2))).unsqueeze(1).float()    # [Batch, 1, Audio_ct]

        x = self.prenet(latents)  # [Batch, Diffusion_d, Audio_ct]
        encodings = self.encoding_ffn(encodings) # [Batch, Diffusion_d, Audio_ct]
        diffusion_steps = self.step_ffn(diffusion_steps) # [Batch, Diffusion_d, 1]

        speech_prompts = self.pre_attention(
            queries= self.pre_attention_query.expand(speech_prompts.size(0), -1, -1),
            keys= speech_prompts,
            values= speech_prompts
            )   # [Batch, Diffusion_d, Token_n]
        
        skips_list = []
        for wavenet in self.wavenets:
            x, skips = wavenet(
                x= x,
                masks= masks,                
                conditions= encodings,
                diffusion_steps= diffusion_steps,
                speech_prompts= speech_prompts
                )   # [Batch, Diffusion_d, Audio_ct]
            skips_list.append(skips)

        x = torch.stack(skips_list, dim= 0).sum(dim= 0) / math.sqrt(self.hp.Diffusion.WaveNet.Stack)
        x = self.postnet(x) * masks

        return x


class Diffusion_Embedding(torch.nn.Module):
    def __init__(
        self,
        channels: int
        ):
        super().__init__()
        self.channels = channels
        self.weight = torch.nn.Parameter(torch.randn(self.channels // 2))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)  # [Batch, 1]
        embeddings = x * self.weight.unsqueeze(0) * 2.0 * math.pi   # [Batch, Dim // 2]
        embeddings = torch.cat([x, embeddings.sin(), embeddings.cos()], dim= 1)    # [Batch, Dim + 1]

        return embeddings

class WaveNet(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        condition_channels: int,
        diffusion_step_channels: int,
        wavenet_dropout_rate: float= 0.0,
        apply_film: bool= False,
        speech_prompt_channels: Optional[int]= None,
        speech_prompt_attention_head: Optional[int]= None
        ):
        super().__init__()
        self.calc_channels = channels
        self.apply_film = apply_film

        self.conv = Conv1d(
            in_channels= channels,
            out_channels= channels * 2,
            kernel_size= kernel_size,
            dilation= dilation,
            padding= (kernel_size - 1) * dilation // 2
            )
        
        self.dropout = torch.nn.Dropout(p= wavenet_dropout_rate)

        self.condition = Conv1d(
            in_channels= condition_channels,
            out_channels= channels * 2,
            kernel_size= 1
            )
        self.diffusion_step = Conv1d(
            in_channels= diffusion_step_channels,
            out_channels= channels,
            kernel_size= 1
            )
        
        if apply_film:
            self.attention = LinearAttention(
                query_channels= channels,
                key_channels= speech_prompt_channels, 
                value_channels= speech_prompt_channels,
                calc_channels= channels,
                num_heads= speech_prompt_attention_head,
                )
            self.film = FilM(
                channels= channels * 2,
                condition_channels= channels,
                )


    def forward(
        self,
        x: torch.FloatTensor,
        masks: torch.FloatTensor,
        conditions: torch.FloatTensor,
        diffusion_steps: torch.FloatTensor,
        speech_prompts: Optional[torch.FloatTensor]
        ):
        residuals = x
        queries = x = x + self.diffusion_step(diffusion_steps)  # [Batch, Calc_d, Time]
        
        x = self.conv(x) + self.condition(conditions)   # [Batch, Calc_d * 2, Time]

        if self.apply_film:
            prompt_conditions = self.attention(
                queries= queries,
                keys= speech_prompts,
                values= speech_prompts,
                )   # [Batch, Diffusion_d, Time]
            x = self.film(x, prompt_conditions, masks)

        x = Fused_Gate(x) # [Batch, Calc_d, Time]
        x = self.dropout(x) * masks # [Batch, Calc_d, Time]

        return x + residuals, x

@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x

class FilM(Conv1d):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        ):
        super().__init__(
            in_channels= condition_channels,
            out_channels= channels * 2,
            kernel_size= 1,
            w_init_gain= 'linear'
            )

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        masks: torch.Tensor,
        ):
        betas, gammas = super().forward(conditions * masks).chunk(chunks= 2, dim= 1)
        x = gammas * x + betas

        return x * masks

def Mask_Generate(lengths: torch.Tensor, max_length: Optional[Union[int, torch.Tensor]]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]

# Chen, T. (2023). On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972.
def sigmoid_schedule(t, start=-3.0, end=3, tau=1.0, clip_min=1e-4):
    # A gamma function based on sigmoid function.
    start = torch.tensor(start)
    end = torch.tensor(end)
    tau = torch.tensor(tau)
    
    v_start = (start / tau).sigmoid()
    v_end = (end / tau).sigmoid()
    output = ((t * (end - start) + start) / tau).sigmoid()
    output = (v_end - output) / (v_end - v_start)
    
    return output.clamp(clip_min, 1.)

