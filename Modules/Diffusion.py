import torch
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

        self.network = Diffusion_Network(
            hyper_parameters= self.hp
            )

        scale = 1000 / self.hp.Diffusion.Max_Step
        beta_start, beta_end = scale * 0.0001, scale * 0.02
        betas = torch.linspace(beta_start, beta_end, self.hp.Diffusion.Max_Step, dtype=torch.double)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis= 0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())  # [Diffusion_t]
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())  # [Diffusion_t]
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt().float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt().float())
        self.register_buffer('sqrt_recip_alphas_cumprod', (1.0 / alphas_cumprod).sqrt().float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', (1.0 / alphas_cumprod - 1.0).sqrt().float())

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance', torch.maximum(posterior_variance, torch.tensor([1e-20])).log().float())
        self.register_buffer('posterior_mean_coef1', (betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)).float())
        self.register_buffer('posterior_mean_coef2', ((1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)).float())

    def forward(
        self,
        features: torch.FloatTensor,
        encodings: torch.FloatTensor,
        lengths: torch.LongTensor,
        speech_prompts: torch.FloatTensor
        ):
        noises = torch.randn_like(features)

        diffusion_steps = torch.randint(
            low= 0,
            high= self.hp.Diffusion.Max_Step,
            size= (encodings.size(0),),
            dtype= torch.long,
            device= encodings.device
            )    # random single step

        noised_features = \
            features * self.sqrt_alphas_cumprod[diffusion_steps][:, None, None] + \
            noises * self.sqrt_one_minus_alphas_cumprod[diffusion_steps][:, None, None]

        outputs = self.network(
            features= noised_features,
            encodings= encodings,
            lengths= lengths,
            speech_prompts= speech_prompts,
            diffusion_steps= diffusion_steps
            )
        
        # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L328-L349
        if self.hp.Diffusion.Network_Prediction.upper() == 'EPSILON':
            epsilons = outputs
            starts = \
                noised_features * self.sqrt_recip_alphas_cumprod[diffusion_steps][:, None, None] - \
                epsilons * self.sqrt_recipm1_alphas_cumprod[diffusion_steps][:, None, None]
        elif self.hp.Diffusion.Network_Prediction.upper() == 'START':
            starts = outputs
            epsilons = \
                (noised_features * self.sqrt_recip_alphas_cumprod[diffusion_steps][:, None, None] - starts) / \
                self.sqrt_recipm1_alphas_cumprod[diffusion_steps][:, None, None]
        else:
            raise NotImplementedError(f'Unknown diffusion network prediction: {self.hp.Diffusion.Network_Prediction}')
        
        return noises, epsilons, starts

    def DDPM(
        self,
        encodings: torch.Tensor,
        lengths: torch.LongTensor,
        speech_prompts: torch.FloatTensor,
        temperature: Optional[float]= 1.2 ** 2
        ):
        features = torch.randn(
            size= (encodings.size(0), self.hp.Audio_Codec_Size, encodings.size(2)),
            device= encodings.device
            )
        for diffusion_steps in reversed(range(self.hp.Diffusion.Max_Step)):
            diffusion_steps =torch.full(
                size= (encodings.size(0), ),
                fill_value= diffusion_steps,
                dtype= torch.long,
                device= encodings.device
                )
            
            outputs = self.network(
                features= features,
                encodings= encodings,
                lengths= lengths,
                speech_prompts= speech_prompts,
                diffusion_steps= diffusion_steps
                )

            if self.hp.Diffusion.Network_Prediction.upper() == 'EPSILON':            
                starts = \
                    features * self.sqrt_recip_alphas_cumprod[diffusion_steps][:, None, None] - \
                    outputs * self.sqrt_recipm1_alphas_cumprod[diffusion_steps][:, None, None]
            elif self.hp.Diffusion.Network_Prediction.upper() == 'START':
                starts = outputs
            else:
                raise NotImplementedError(f'Unknown diffusion network prediction: {self.hp.Diffusion.Network_Prediction}')
            starts.clamp_(-1.0, 1.0)  # clipped

            posterior_means = \
                starts * self.posterior_mean_coef1[diffusion_steps][:, None, None] + \
                features * self.posterior_mean_coef2[diffusion_steps][:, None, None]
            posterior_log_variances = \
                self.posterior_log_variance[diffusion_steps][:, None, None]
            
            noises = torch.randn_like(features) / temperature # [Batch, Feature_d, Feature_d]
            masks = (diffusion_steps > 0).float().unsqueeze(1).unsqueeze(1) # [Batch, 1, 1]
            features = posterior_means + masks * (0.5 * posterior_log_variances).exp() * noises
        
        return features

    # def DDIM(
    #     self,
    #     encodings: torch.Tensor,
    #     lengths: torch.LongTensor,
    #     speech_prompts: torch.FloatTensor,
    #     ddim_steps: int,
    #     eta: float= 0.0,
    #     temperature: Optional[float]= 1.2 ** 2
    #     ):
    #     ddim_timesteps = self.Get_DDIM_Steps(
    #         ddim_steps= ddim_steps
    #         )

    #     features = torch.randn(
    #         size= (encodings.size(0), self.hp.Audio_Codec_Size, encodings.size(2)),
    #         device= encodings.device
    #         )

    #     for diffusion_steps in reversed(ddim_timesteps):
    #         diffusion_steps = torch.full(
    #             size= (encodings.size(0), ),
    #             fill_value= diffusion_steps,
    #             dtype= torch.long,
    #             device= encodings.device
    #             )

    #         outputs = self.network(
    #             features= features,
    #             encodings= encodings,
    #             lengths= lengths,
    #             speech_prompts= speech_prompts,
    #             diffusion_steps= diffusion_steps
    #             )

    #         if self.hp.Diffusion.Network_Prediction.upper() == 'EPSILON':            
    #             epsilons = outputs
    #             starts = \
    #                 features * self.sqrt_recip_alphas_cumprod[diffusion_steps][:, None, None] - \
    #                 outputs * self.sqrt_recipm1_alphas_cumprod[diffusion_steps][:, None, None]
    #         elif self.hp.Diffusion.Network_Prediction.upper() == 'START':
    #             starts = outputs
    #             epsilons = \
    #                 (noised_features * self.sqrt_recip_alphas_cumprod[diffusion_steps][:, None, None] - starts) / \
    #                 self.sqrt_recipm1_alphas_cumprod[diffusion_steps][:, None, None]
    #         else:
    #             raise NotImplementedError(f'Unknown diffusion network prediction: {self.hp.Diffusion.Network_Prediction}')
    #         starts.clamp_(-1.0, 1.0)  # clipped

    #         alphas_cumprod = self.alphas_cumprod[diffusion_steps][:, None, None]
    #         alphas_cumprod_prev = self.alphas_cumprod_prev[diffusion_steps][:, None, None]
    #         sigmas = eta * ((1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).sqrt() * ((1.0 - alphas_cumprod) / alphas_cumprod_prev).sqrt()

    #         noises = torch.randn_like(features) / temperature # [Batch, Feature_d, Feature_d]
    #         masks = (diffusion_steps > 0).float().unsqueeze(1).unsqueeze(1) # [Batch, 1, 1]
    #         features = starts * alphas_cumprod_prev.sqrt() + (1.0 - alphas_cumprod_prev - sigmas ** 2).sqrt() * epsilons
    #         features = features + masks * sigmas * noises

    #     return features

    # # https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
    # def Get_DDIM_Steps(
    #     self,        
    #     ddim_steps: int,
    #     ddim_discr_method: str= 'uniform'
    #     ):
    #     if ddim_discr_method == 'uniform':            
    #         ddim_timesteps = torch.arange(0, self.hp.Diffusion.Max_Step, self.hp.Diffusion.Max_Step // ddim_steps).long()
    #     elif ddim_discr_method == 'quad':
    #         ddim_timesteps = torch.linspace(0, (torch.tensor(self.hp.Diffusion.Max_Step) * 0.8).sqrt(), ddim_steps).pow(2.0).long()
    #     else:
    #         raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        
    #     ddim_timesteps[-1] = self.hp.Diffusion.Max_Step - 1

    #     return list(reversed(ddim_timesteps))

    def DDIM(
        self,
        encodings: torch.Tensor,
        lengths: torch.LongTensor,
        speech_prompts: torch.FloatTensor,
        ddim_steps: int,
        eta: float= 0.0,
        temperature: Optional[float]= 1.2 ** 2
        ):
        ddim_timesteps = self.Get_DDIM_Steps(
            ddim_steps= ddim_steps
            )
        sigmas, alphas, alphas_prev = self.Get_DDIM_Sampling_Parameters(
            ddim_timesteps= ddim_timesteps,
            eta= eta
            )
        sqrt_one_minus_alphas = (1. - alphas).sqrt()

        features = torch.randn(
            size= (encodings.size(0), self.hp.Audio_Codec_Size, encodings.size(2)),
            device= encodings.device
            )

        setp_range = reversed(range(ddim_steps))

        for diffusion_steps in setp_range:
            noised_predictions = self.network(
                features= features,
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

            feature_starts = (features - sqrt_one_minus_alphas[diffusion_steps] * noised_predictions) / alphas[diffusion_steps].sqrt()
            direction_pointings = (1.0 - alphas_prev[diffusion_steps] - sigmas[diffusion_steps].pow(2.0)) * noised_predictions
            noises = sigmas[diffusion_steps] * torch.randn_like(features) / temperature

            features = alphas_prev[diffusion_steps].sqrt() * feature_starts + direction_pointings + noises

        return features

    # https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
    def Get_DDIM_Steps(
        self,        
        ddim_steps: int,
        ddim_discr_method: str= 'uniform'
        ):
        if ddim_discr_method == 'uniform':            
            ddim_timesteps = torch.arange(0, self.hp.Diffusion.Max_Step, self.hp.Diffusion.Max_Step // ddim_steps).long()
        elif ddim_discr_method == 'quad':
            ddim_timesteps = torch.linspace(0, (torch.tensor(self.hp.Diffusion.Max_Step) * 0.8).sqrt(), ddim_steps).pow(2.0).long()
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        
        ddim_timesteps[-1] = self.hp.Diffusion.Max_Step - 1

        return ddim_timesteps

    def Get_DDIM_Sampling_Parameters(self, ddim_timesteps, eta):
        alphas = self.alphas_cumprod[ddim_timesteps]
        alphas_prev = self.alphas_cumprod_prev[ddim_timesteps]
        sigmas = eta * ((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)).sqrt()

        return sigmas, alphas, alphas_prev

class Diffusion_Network(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Audio_Codec_Size,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.SiLU()
            )

        self.encoding_ffn = torch.nn.Sequential(
            Conv1d(
                in_channels= self.hp.Encoder.Size + self.hp.Audio_Codec_Size,
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
                speech_prompt_channels= self.hp.Diffusion.Pre_Attention.Query_Size,
                speech_prompt_attention_head= self.hp.Diffusion.WaveNet.Attention.Head
                )
            for wavenet_index in range(self.hp.Diffusion.WaveNet.Stack)
            ])

        self.postnet = torch.nn.Sequential(
            torch.nn.SiLU(),
            Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Audio_Codec_Size,
                kernel_size= 1,
                w_init_gain= 'zero'
                )
            ) 

    def forward(
        self,
        features: torch.FloatTensor,
        encodings: torch.FloatTensor,
        lengths: torch.LongTensor,
        speech_prompts: torch.FloatTensor,
        diffusion_steps: torch.Tensor,
        ):
        '''
        latents: [Batch, Codec_d, Audio_ct]
        encodings: [Batch, Enc_d, Audio_ct]
        diffusion_steps: [Batch]
        speech_prompts: [Batch, Prompt_d, Prompt_t]
        '''
        masks= (~Mask_Generate(lengths, max_length= features.size(2))).unsqueeze(1).float()    # [Batch, 1, Audio_ct]
        diffusion_steps = diffusion_steps.float() / self.hp.Diffusion.Max_Step

        x = self.prenet(features)  # [Batch, Diffusion_d, Audio_ct]
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