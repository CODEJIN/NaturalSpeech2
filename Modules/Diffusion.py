import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm

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

        self.timesteps = self.hp.Diffusion.Max_Step
        betas = torch.linspace(1e-4, 0.06, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis= 0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('alphas_cumprod', alphas_cumprod)  # [Diffusion_t]
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # [Diffusion_t]
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recip_alphas_cumprod', (1.0 / alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', (1.0 / alphas_cumprod - 1.0).sqrt())

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance', torch.maximum(posterior_variance, torch.tensor([1e-20])).log())
        self.register_buffer('posterior_mean_coef1', betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod))

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
            diffusion_steps = torch.randint(
                low= 0,
                high= self.timesteps,
                size= (encodings.size(0),),
                dtype= torch.long,
                device= encodings.device
                )    # random single step
            
            noises, epsilons = self.Get_Noise_Epsilon_for_Train(
                latents= latents,
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts,
                diffusion_steps= diffusion_steps,
                )
            return None, noises, epsilons
        else:   # inference
            latents = self.Sampling(
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts,
                )
            return latents, None, None

    def Sampling(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.FloatTensor,
        ):
        latents = torch.randn(
            size= (encodings.size(0), self.hp.Audio_Codec.Size, encodings.size(2)),
            device= encodings.device
            )
        for diffusion_step in reversed(range(self.timesteps)):
            latents = self.P_Sampling(
                latents= latents,
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts,
                diffusion_steps= torch.full(
                    size= (encodings.size(0), ),
                    fill_value= diffusion_step,
                    dtype= torch.long,
                    device= encodings.device
                    ),
                )
        
        return latents

    def P_Sampling(
        self,
        latents: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.FloatTensor,
        diffusion_steps: torch.Tensor,
        ):
        posterior_means, posterior_log_variances = self.Get_Posterior(
            latents= latents,
            encodings= encodings,
            lengths= lengths,
            speech_prompts= speech_prompts,
            diffusion_steps= diffusion_steps,
            )

        noises = torch.randn_like(latents) # [Batch, Feature_d, Feature_d]
        masks = (diffusion_steps > 0).float().unsqueeze(1).unsqueeze(1) #[Batch, 1, 1]
        
        return posterior_means + masks * (0.5 * posterior_log_variances).exp() * noises

    def Get_Posterior(
        self,
        latents: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.FloatTensor,
        diffusion_steps: torch.Tensor
        ):
        noised_predictions = self.denoiser(
            latents= latents,
            lengths= lengths,
            encodings= encodings,
            speech_prompts= speech_prompts,
            diffusion_steps= diffusion_steps
            )

        epsilons = \
            latents * self.sqrt_recip_alphas_cumprod[diffusion_steps][:, None, None] - \
            noised_predictions * self.sqrt_recipm1_alphas_cumprod[diffusion_steps][:, None, None]
        epsilons.clamp_(-1.0, 1.0)  # clipped
        
        posterior_means = \
            epsilons * self.posterior_mean_coef1[diffusion_steps][:, None, None] + \
            latents * self.posterior_mean_coef2[diffusion_steps][:, None, None]
        posterior_log_variances = \
            self.posterior_log_variance[diffusion_steps][:, None, None]

        return posterior_means, posterior_log_variances

    def Get_Noise_Epsilon_for_Train(
        self,
        latents: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.FloatTensor,
        diffusion_steps: torch.Tensor,
        ):
        noises = torch.randn_like(latents)

        noised_latents = \
            latents * self.sqrt_alphas_cumprod[diffusion_steps][:, None, None] + \
            noises * self.sqrt_one_minus_alphas_cumprod[diffusion_steps][:, None, None]

        epsilons = self.denoiser(
            latents= noised_latents,
            encodings= encodings,
            lengths= lengths,
            speech_prompts= speech_prompts,
            diffusion_steps= diffusion_steps,
            )
        
        return noises, epsilons

    def DDIM(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.FloatTensor,
        ddim_steps: int,
        eta: float= 0.0,
        temperature: float= 1.0,
        use_tqdm: bool= False
        ):
        ddim_timesteps = self.Get_DDIM_Steps(
            ddim_steps= ddim_steps
            )
        sigmas, alphas, alphas_prev = self.Get_DDIM_Sampling_Parameters(
            ddim_timesteps= ddim_timesteps,
            eta= eta
            )
        sqrt_one_minus_alphas = (1. - alphas).sqrt()

        latents = torch.randn(
            size= (encodings.size(0), self.hp.Audio_Codec.Size, encodings.size(2)),
            device= encodings.device
            )

        setp_range = reversed(range(ddim_steps))
        if use_tqdm:
            tqdm(
                setp_range,
                desc= '[Diffusion]',
                total= ddim_steps
                )

        for diffusion_steps in setp_range:
            noised_predictions = self.denoiser(
                latents= latents,
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts,
                diffusion_steps= torch.full(
                    size= (encodings.size(0), ),
                    fill_value= diffusion_steps,
                    dtype= torch.long,
                    device= encodings.device
                    )
                )

            audio_codec_encoding_starts = (latents - sqrt_one_minus_alphas[diffusion_steps] * noised_predictions) / alphas[diffusion_steps].sqrt()
            direction_pointings = (1.0 - alphas_prev[diffusion_steps] - sigmas[diffusion_steps].pow(2.0)) * noised_predictions
            noises = sigmas[diffusion_steps] * torch.randn_like(latents) * temperature

            latents = alphas_prev[diffusion_steps].sqrt() * audio_codec_encoding_starts + direction_pointings + noises

        return latents

    # https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
    def Get_DDIM_Steps(
        self,        
        ddim_steps: int,
        ddim_discr_method: str= 'uniform'
        ):
        if ddim_discr_method == 'uniform':            
            ddim_timesteps = torch.arange(0, self.timesteps, self.timesteps // ddim_steps).long()
        elif ddim_discr_method == 'quad':
            ddim_timesteps = torch.linspace(0, (torch.tensor(self.timesteps) * 0.8).sqrt(), ddim_steps).pow(2.0).long()
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        
        ddim_timesteps[-1] = self.timesteps - 1

        return ddim_timesteps

    def Get_DDIM_Sampling_Parameters(self, ddim_timesteps, eta):
        alphas = self.alphas_cumprod[ddim_timesteps]
        alphas_prev = self.alphas_cumprod_prev[ddim_timesteps]
        sigmas = eta * ((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)).sqrt()

        return sigmas, alphas, alphas_prev

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
            torch.nn.ReLU()
            )

        self.encoding_ffn = torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.Encoder.Size * 4,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.ReLU(),
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
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Diffusion.Size * 4,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.ReLU(),
            Conv1d(
                in_channels= self.hp.Diffusion.Size * 4,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                w_init_gain= 'linear'
                )
            )

        self.wavenets = torch.nn.ModuleList([
            WaveNet(
                channels= self.hp.Diffusion.Size,
                conv_stack= self.hp.Diffusion.WaveNet.Conv_Stack,
                kernel_size= self.hp.Diffusion.WaveNet.Kernel_Size,
                dilation_rate= self.hp.Diffusion.WaveNet.Dilation_Rate,
                dropout_rate= self.hp.Diffusion.WaveNet.Dropout_Rate,
                condition_channels= self.hp.Diffusion.Size,
                )
            for _ in range(self.hp.Diffusion.Stack)
            ])
        
        self.pre_attention = torch.nn.MultiheadAttention(
            embed_dim= self.hp.Diffusion.Attention.Query_Size,
            num_heads= self.hp.Diffusion.Attention.Head,
            dropout= self.hp.Diffusion.Attention.Dropout_Rate,
            kdim= self.hp.Speech_Prompter.Size,
            vdim= self.hp.Speech_Prompter.Size
            )
        self.pre_attention_query = torch.nn.Parameter(
            torch.randn(self.hp.Diffusion.Attention.Query_Token, 1, self.hp.Diffusion.Attention.Query_Size)
            )
        
        self.attentions = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(
                embed_dim= self.hp.Diffusion.Size,
                num_heads= self.hp.Diffusion.Attention.Head,
                dropout= self.hp.Diffusion.Attention.Dropout_Rate,
                kdim= self.hp.Speech_Prompter.Size,
                vdim= self.hp.Speech_Prompter.Size
                )
            for _ in range(self.hp.Diffusion.Stack)
            ])

        self.films = torch.nn.ModuleList([
            FilM(
                channels= self.hp.Diffusion.Size,
                condition_channels= self.hp.Diffusion.Size,
                )
            for _ in range(self.hp.Diffusion.Stack)
            ])
        
        self.postnet = Conv1d(
            in_channels= self.hp.Diffusion.Size,
            out_channels= self.hp.Audio_Codec.Size,
            kernel_size= 1,
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
        Diffusion_d == Codec_d * 2
        '''
        masks= (~Mask_Generate(lengths, max_length= latents.size(2))).unsqueeze(1).float()    # [Batch, 1, Audio_ct]

        x = self.prenet(latents)  # [Batch, Diffusion_d, Audio_ct]
        encodings = self.encoding_ffn(encodings) # [Batch, Diffusion_d, Audio_ct]
        diffusion_steps = self.step_ffn(diffusion_steps) # [Batch, Diffusion_d, 1]
        conditions = encodings + diffusion_steps    # [Batch, Diffusion_d, Audio_ct]

        speech_prompts = self.pre_attention(
            query= self.pre_attention_query.expand(-1, speech_prompts.size(0), -1),
            key= speech_prompts.permute(2, 0, 1),
            value= speech_prompts.permute(2, 0, 1),
            )[0]   # [Token_n, Batch, Diffusion_d]

        for wavenet, attention, film in zip(self.wavenets, self.attentions, self.films):
            x = wavenet(
                x= x,
                masks= masks,
                conditions= conditions
                )   # [Batch, Diffusion_d, Audio_ct]            
            prompt_conditions = attention(
                query= x.permute(2, 0, 1),
                key= speech_prompts,
                value= speech_prompts,
                )[0].permute(1, 2, 0)   # [Batch, Diffusion_d, Audio_ct]            
            x = film(x, prompt_conditions, masks)   # [Batch, Diffusion_d, Audio_ct]

        x = self.postnet(x) * masks

        return x
    
class Diffusion_Embedding(torch.nn.Module):
    def __init__(
        self,
        channels: int
        ):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor):
        half_channels = self.channels // 2  # sine and cosine
        embeddings = math.log(10000.0) / (half_channels - 1)
        embeddings = torch.exp(torch.arange(half_channels, device= x.device) * -embeddings)
        embeddings = x.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim= -1)

        return embeddings

class WaveNet(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        conv_stack: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float= 0.0,
        condition_channels: Optional[int]= None,
        ):
        super().__init__()
        self.channels = channels
        self.conv_stack = conv_stack
        self.use_condition = not condition_channels is None

        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)

        if self.use_condition:
            self.condition = torch.nn.utils.weight_norm(Conv1d(
                in_channels= condition_channels,
                out_channels= channels * conv_stack * 2,
                kernel_size= 1
                ))
            self.condition.apply(weight_norm_initialize_weight)
        
        self.input_convs = torch.nn.ModuleList()
        self.residual_and_skip_convs = torch.nn.ModuleList()
        for index in range(conv_stack):
            dilation = dilation_rate ** index
            padding = (kernel_size - 1) * dilation // 2
            self.input_convs.append(torch.nn.utils.weight_norm(Conv1d(
                in_channels= channels,
                out_channels= channels * 2,
                kernel_size= kernel_size,
                dilation= dilation,
                padding= padding
                )))
            self.residual_and_skip_convs.append(torch.nn.utils.weight_norm(Conv1d(
                in_channels= channels,
                out_channels= channels * 2,
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
                size= (x.size(0), self.channels * 2, x.size(2)),
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

class FilM(Conv1d):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        ):
        super().__init__(
            in_channels= condition_channels,
            out_channels= channels * 2,
            kernel_size= 1
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
