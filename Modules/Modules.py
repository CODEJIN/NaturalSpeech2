from argparse import Namespace
import torch, torchaudio, torchvision
import math
from typing import Optional, List, Dict, Tuple, Union

from .Diffusion import Diffusion
from .Codec import Encoder as Audio_Codec_Encoder, Decoder as Audio_Codec_Decoder
from .Meta_RVQ import ResidualVectorQuantization
from .Layer import Conv1d, ConvTranspose1d, Lambda, LayerNorm
from meldataset import spectrogram_to_mel, mel_spectrogram

class NaturalSpeech2(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.text_encoder = Phoneme_Encoder(self.hp)

        self.speech_prompter = Speech_Prompter(self.hp)

        self.duration_predictor = Duration_Predictor(self.hp)
        self.f0_predictor = F0_Predictor(self.hp)

        self.diffusion = Diffusion(self.hp)

        self.audio_codec_encoder = Audio_Codec_Encoder(self.hp)
        self.residual_vq = ResidualVectorQuantization(
            num_quantizers= self.hp.Audio_Codec.Residual_VQ.Stack,
            dim= self.hp.Audio_Codec.Size,
            codebook_size= self.hp.Audio_Codec.Residual_VQ.Num_Codebook,
            codebook_dim= self.hp.Audio_Codec.Size,
            )
        self.audio_codec_decoder = Audio_Codec_Decoder(self.hp)
        
        self.segment = Segment()

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        ge2es: torch.FloatTensor,
        features: Optional[torch.FloatTensor]= None,
        feature_lengths: Optional[torch.Tensor]= None,
        f0s: Optional[torch.Tensor]= None,
        audios: Optional[torch.Tensor]= None,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0,
        ):
        if not features is None and not feature_lengths is None:    # train
            return self.Train(
                tokens= tokens,
                token_lengths= token_lengths,
                ge2es= ge2es,
                features= features,
                feature_lengths= feature_lengths,
                f0s= f0s,
                audios= audios
                )
        else:   #  inference
            return self.Inference(
                tokens= tokens,
                token_lengths= token_lengths,
                ge2es= ge2es,
                length_scales= length_scales
                )

    def Train(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        ge2es: torch.FloatTensor,
        features: torch.FloatTensor,
        feature_lengths: torch.Tensor,
        f0s: torch.Tensor,
        audios: torch.Tensor
        ):
        encoding_means, encoding_log_stds, encodings = self.text_encoder(
            tokens= tokens,
            lengths= token_lengths
            )
        conditions = self.condition(ge2es)
        
        linguistic_means, linguistic_log_stds = self.linguistic_encoder(audios, feature_lengths)
        linguistic_samples = linguistic_means + linguistic_log_stds.exp() * torch.randn_like(linguistic_log_stds)
        linguistic_flows = self.linguistic_flow(
            x= linguistic_samples,
            lengths= feature_lengths,
            conditions= conditions,
            reverse= False
            )   # [Batch, Enc_d, Feature_t]

        acoustic_means, acoustic_log_stds = self.acoustic_encoder(features, feature_lengths)
        acoustic_samples = acoustic_means + acoustic_log_stds.exp() * torch.randn_like(acoustic_log_stds)
        acoustic_flows = self.acoustic_flow(
            x= acoustic_samples,
            lengths= feature_lengths,
            conditions= conditions,
            reverse= False
            )   # [Batch, Enc_d, Feature_t]

        durations, alignments = Calc_Duration(
            encoding_means= encoding_means,
            encoding_log_stds= encoding_log_stds,
            encoding_lengths= token_lengths,
            decodings= linguistic_flows,
            decoding_lengths= feature_lengths,
            )
        _, duration_losses = self.variance_block(
            encodings= encodings,
            encoding_lengths= token_lengths,
            durations= durations
            )

        encoding_means = encoding_means @ alignments.permute(0, 2, 1)
        encoding_log_stds = encoding_log_stds @ alignments.permute(0, 2, 1)
        encoding_samples = encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds)
        f0_predictions, f0_embeddings = self.f0_predictor(
            encodings= encoding_samples,
            lengths= feature_lengths,
            f0s= f0s
            )

        acoustic_samples_slice, offsets = self.segment(
            patterns= (acoustic_samples + f0_embeddings).permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            lengths= feature_lengths
            )
        acoustic_samples_slice = acoustic_samples_slice.permute(0, 2, 1)    # [Batch, Enc_d, Feature_st]

        mels = spectrogram_to_mel(
            features,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            win_size= self.hp.Sound.Frame_Length,
            fmin= 0,
            fmax= None,
            use_denorm= False
            )
        mels_slice, _ = self.segment(
            patterns= mels.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            offsets= offsets
            )
        mels_slice = mels_slice.permute(0, 2, 1)    # [Batch, Mel_d, Feature_st]
        
        audios_slice, _ = self.segment(
            patterns= audios,
            segment_size= self.hp.Train.Segment_Size * self.hp.Sound.Frame_Shift,
            offsets= offsets * self.hp.Sound.Frame_Shift
            )   # [Batch, Audio_st(Feature_st * Hop_Size)]

        audio_predictions_slice = self.decoder(
            encodings= acoustic_samples_slice,
            lengths= torch.full_like(feature_lengths, self.hp.Train.Segment_Size),
            conditions= conditions,
            )

        mel_predictions_slice = mel_spectrogram(
            audio_predictions_slice,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Frame_Shift,
            win_size= self.hp.Sound.Frame_Length,
            fmin= 0,
            fmax= None
            )

        token_predictions = self.token_predictor(
            encodings= linguistic_samples
            )


        
        
        # forwad flow from natural speech
        linguistic_samples_forward = self.linguistic_flow(
            x= encoding_samples,
            lengths= feature_lengths,
            conditions= conditions,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]        
        acoustic_samples_forward = self.acoustic_flow(
            x= linguistic_samples,
            lengths= feature_lengths,
            conditions= conditions,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]
        
        acoustic_samples_forward_slice, offsets = self.segment(
            patterns= (acoustic_samples_forward + f0_embeddings).permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            lengths= feature_lengths
            )
        acoustic_samples_forward_slice = acoustic_samples_forward_slice.permute(0, 2, 1)    # [Batch, Enc_d, Feature_st]

        audios_forward_slice, _ = self.segment(
            patterns= audios,
            segment_size= self.hp.Train.Segment_Size * self.hp.Sound.Frame_Shift,
            offsets= offsets * self.hp.Sound.Frame_Shift
            )   # [Batch, Audio_st(Feature_st * Hop_Size)]

        audio_predictions_forward_slice = self.decoder(
            encodings= acoustic_samples_forward_slice,
            lengths= torch.full_like(feature_lengths, self.hp.Train.Segment_Size),
            conditions= conditions,
            )

        return \
            audio_predictions_slice, audios_slice, mel_predictions_slice, mels_slice, \
            audio_predictions_forward_slice, audios_forward_slice, \
            encoding_means, encoding_log_stds, linguistic_flows, linguistic_log_stds, \
            linguistic_means, linguistic_log_stds, linguistic_samples_forward, encoding_log_stds, \
            linguistic_means, linguistic_log_stds, acoustic_flows, acoustic_log_stds, \
            acoustic_means, acoustic_log_stds, acoustic_samples_forward, linguistic_log_stds, \
            duration_losses, token_predictions, f0_predictions, alignments

    def Inference(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        ge2es: torch.FloatTensor,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0
        ):
        length_scales = self.Scale_to_Tensor(tokens= tokens, scale= length_scales)

        encoding_means, encoding_log_stds, encodings = self.text_encoder(
            tokens= tokens,
            lengths= token_lengths
            )
        conditions = self.condition(ge2es)
        
        alignments, _ = self.variance_block(
            encodings= encodings,
            encoding_lengths= token_lengths,
            length_scales= length_scales
            )
        feature_lengths = alignments.sum(dim= [1, 2])
        
        encoding_means = encoding_means @ alignments.permute(0, 2, 1)
        encoding_log_stds = encoding_log_stds @ alignments.permute(0, 2, 1)
        encoding_samples = encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds)
        f0_predictions, f0_embeddings = self.f0_predictor(
            encodings= encoding_samples,
            lengths= feature_lengths
            )

        linguistic_samples = self.linguistic_flow(
            x= encoding_samples,
            lengths= feature_lengths,
            conditions= conditions,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]

        acoustic_samples = self.acoustic_flow(
            x= linguistic_samples,
            lengths= feature_lengths,
            conditions= conditions,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]

        audio_predictions = self.decoder(
            encodings= acoustic_samples + f0_embeddings,
            lengths= feature_lengths,
            conditions= conditions,
            )

        return \
            audio_predictions, None, None, None, \
            None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, f0_predictions, alignments

    def Scale_to_Tensor(
        self,
        tokens: torch.Tensor,
        scale: Union[float, List[float], torch.Tensor]
        ):
        if isinstance(scale, float):
            scale = torch.FloatTensor([scale,]).unsqueeze(0).expand(tokens.size(0), tokens.size(1))
        elif isinstance(scale, list):
            if len(scale) != tokens.size(0):
                raise ValueError(f'When scale is a list, the length must be same to the batch size: {len(scale)} != {tokens.size(0)}')
            scale = torch.FloatTensor(scale).unsqueeze(1).expand(tokens.size(0), tokens.size(1))
        elif isinstance(scale, torch.Tensor):
            if scale.ndim != 2:
                raise ValueError('When scale is a tensor, ndim must be 2.')
            elif scale.size(0) != tokens.size(0):
                raise ValueError(f'When scale is a tensor, the dimension 0 of tensor must be same to the batch size: {scale.size(0)} != {tokens.size(0)}')
            elif scale.size(1) != tokens.size(1):
                raise ValueError(f'When scale is a tensor, the dimension 1 of tensor must be same to the token length: {scale.size(1)} != {tokens.size(1)}')

        return scale.to(tokens.device)



class Phoneme_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size,
            )
        embedding_variance = math.sqrt(3.0) * math.sqrt(2.0 / (self.hp.Tokens + self.hp.Encoder.Size))
        self.token_embedding.weight.data.uniform_(-embedding_variance, embedding_variance)

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Transformer.Head,
                feedforward_kernel_size= self.hp.Encoder.Transformer.FFN.Kernel_Size,
                dropout_rate= self.hp.Encoder.Transformer.Dropout_Rate,
                feedforward_dropout_rate= self.hp.Encoder.Transformer.FFN.Dropout_Rate,
                )
            for index in range(self.hp.Encoder.Transformer.Stack)
            ])

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        ) -> torch.Tensor:
        '''
        tokens: [Batch, Time]
        '''
        encodings = self.token_embedding(tokens).permute(0, 2, 1)
        
        for block in self.blocks:
            encodings = block(encodings, lengths)
        
        return encodings

class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,
        feedforward_kernel_size: int,
        dropout_rate: float= 0.0,
        feedforward_dropout_rate: float= 0.2
        ) -> None:
        super().__init__()

        self.positional_encoding = Positional_Encoding(channels)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim= channels,
            num_heads= num_head,
            dropout= dropout_rate
            )
        
        self.ffn = FFN(
            channels= channels,
            kernel_size= feedforward_kernel_size,
            dropout_rate= feedforward_dropout_rate
            )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())   # [Batch, Time]

        x = self.positional_encoding(x).permute(2, 0, 1)    # [Time, Batch, Dim]
        
        x = self.attention(
            query= x,
            key= x,
            value= x,
            key_padding_mask= masks
            ).permute(1, 2, 0)
        
        # FFN + Dropout + LayerNorm
        masks = (~masks).unsqueeze(1).float()   # float mask
        x = self.ffn(x, masks)

        return x * masks

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout_rate: float= 0.1,
        ) -> None:
        super().__init__()
        self.conv_0 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'relu'
            )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        self.norm = LayerNorm(
            num_features= channels,
            )
        
    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        residuals = x

        x = self.conv_0(x * masks)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_1(x * masks)
        x = self.dropout(x)
        x = self.norm(x + residuals)

        return x * masks


class Speech_Prompter(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Speech_Prompter.Size,
                num_head= self.hp.Speech_Prompter.Transformer.Head,
                feedforward_kernel_size= self.hp.Speech_Prompter.Transformer.FFN.Kernel_Size,
                dropout_rate= self.hp.Speech_Prompter.Transformer.Dropout_Rate,
                feedforward_dropout_rate= self.hp.Speech_Prompter.Transformer.FFN.Dropout_Rate,
                )
            for index in range(self.hp.Speech_Prompter.Transformer.Stack)
            ])

    def forward(
        self,
        sources: torch.Tensor,
        lengths: torch.Tensor,
        ) -> torch.Tensor:
        '''
        tokens: [Batch, Time]
        '''
        
        for block in self.blocks:
            encodings = block(encodings, lengths)
        
        return encodings


class Variance_Predictor(torch.nn.Module): 
    def __init__(
        self,
        channels: int,
        stack: int,
        attention_num_head: int,
        attention_dropout_rate: float,        
        conv_kernel_size: int,
        conv_stack_in_stack: int,
        conv_dropout_rate: float
        ):
        super().__init__()
        
        self.conv_blocks = torch.nn.ModuleList()
        for index in range(stack):
            conv_block = torch.nn.ModuleList()
            for conv_block_index in range(conv_stack_in_stack):
                conv_block.append(Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= conv_kernel_size,
                    padding= (conv_kernel_size - 1) // 2
                    ))
                conv_block.append(LayerNorm(num_features= channels))
                conv_block.append(torch.nn.ReLU())
                conv_block.append(torch.nn.Dropout(p= conv_dropout_rate))
                self.conv_blocks.append(conv_block)

        self.attentions = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(
                embed_dim= channels,
                num_heads= attention_num_head,
                dropout= attention_dropout_rate
                )
            for index in range(stack)
            ])

    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.Tensor
        ) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Enc_t]
        speech_prompts: [Batch, Enc_d, Prompt_t]
        '''
        masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum()) # [Batch, Enc_t]
        float_masks = (~masks).unsqueeze(1).float()   # float mask, [Batch, 1, Enc_t]
        x = encodings

        for conv_blocks, attention in zip(self.conv_blocks, self.attentions):
            for block in conv_blocks:
                x = block(x * float_masks) * float_masks

            residuals = x            
            x = x.permute(2, 0, 1)
            speech_prompts = speech_prompts.permute(2, 0, 1)
            x = attention(
                query= x,
                key= speech_prompts,
                value= speech_prompts,
                key_padding_mask= masks
                ).permute(1, 2, 0)
            
            x = x + residuals

        return x * masks

class Duration_Predictor(Variance_Predictor):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        self.hp = hyper_parameters        
        super().__init__(
            channels= self.hp.Encoder.Size,
            stack= self.hp.Duration_Predictor.Stack,
            attention_num_head= self.hp.Duration_Predictor.Attention.Num_Head,
            attention_dropout_rate= self.hp.Duration_Predictor.Attention.Dropout_Rate,
            conv_kernel_size= self.hp.Duration_Predictor.Conv.Kernel_Size,
            conv_stack_in_stack= self.hp.Duration_Predictor.Conv.Stack,
            conv_dropout_rate= self.hp.Duration_Predictor.Conv.Dropout_Rate,
            )
        
class F0_Predictor(Variance_Predictor):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        self.hp = hyper_parameters        
        super().__init__(
            channels= self.hp.Encoder.Size,
            stack= self.hp.Duration_Predictor.Stack,
            attention_num_head= self.hp.Duration_Predictor.Attention.Num_Head,
            attention_dropout_rate= self.hp.Duration_Predictor.Attention.Dropout_Rate,
            conv_kernel_size= self.hp.Duration_Predictor.Conv.Kernel_Size,
            conv_stack_in_stack= self.hp.Duration_Predictor.Conv.Stack,
            conv_dropout_rate= self.hp.Duration_Predictor.Conv.Dropout_Rate,
            )


class Decoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.condition = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1,
            )
        self.vae_memory_bank = VAE_Memory_Bank(self.hp)

        self.prenet = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Decoder.Upsample.Base_Size,
            kernel_size= self.hp.Decoder.Prenet.Kernel_Size,
            padding= (self.hp.Decoder.Prenet.Kernel_Size - 1) // 2,
            # w_init_gain= 'leaky_relu'   # Don't use this line.
            )

        self.upsample_blocks = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Decoder.Upsample.Base_Size
        for index, (upsample_rate, kernel_size) in enumerate(zip(
            self.hp.Decoder.Upsample.Rate,
            self.hp.Decoder.Upsample.Kernel_Size
            )):
            current_channels = self.hp.Decoder.Upsample.Base_Size // (2 ** (index + 1))
            upsample_block = torch.nn.Sequential(
                torch.nn.LeakyReLU(
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ),
                torch.nn.utils.weight_norm(ConvTranspose1d(
                    in_channels= previous_channels,
                    out_channels= current_channels,
                    kernel_size= kernel_size,
                    stride= upsample_rate,
                    padding= (kernel_size - upsample_rate) // 2
                    ))
                )
            self.upsample_blocks.append(upsample_block)
            residual_blocks = torch.nn.ModuleList()
            for residual_kernel_size, residual_dilation_size in zip(
                self.hp.Decoder.Residual_Block.Kernel_Size,
                self.hp.Decoder.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Decoder_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size,
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        self.postnet = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            Conv1d(
                in_channels= previous_channels,
                out_channels= 1,
                kernel_size= self.hp.Decoder.Postnet.Kernel_Size,
                padding= (self.hp.Decoder.Postnet.Kernel_Size - 1) // 2,
                bias= False,
                # w_init_gain= 'tanh' # Don't use this line.
                ),
            torch.nn.Tanh(),
            Lambda(lambda x: x.squeeze(1))
            )

        # This is critical when using weight normalization.
        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)
        self.upsample_blocks.apply(weight_norm_initialize_weight)
        self.residual_blocks.apply(weight_norm_initialize_weight)
            
    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        conditions: torch.Tensor,
        ) -> torch.Tensor:
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(encodings[0, 0]).sum())).unsqueeze(1).float()
        if conditions.ndim == 2:
            conditions = conditions.unsqueeze(2)

        decodings = self.vae_memory_bank(encodings + self.condition(conditions)) * masks
        decodings = self.prenet(decodings) * masks
        for upsample_block, residual_blocks, upsample_rate in zip(
            self.upsample_blocks,
            self.residual_blocks,
            self.hp.Decoder.Upsample.Rate
            ):
            decodings = upsample_block(decodings)
            lengths = lengths * upsample_rate
            masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(decodings[0, 0]).sum())).unsqueeze(1).float()
            decodings = torch.stack(
                [block(decodings, masks) for block in residual_blocks],
                # [block(decodings) for block in residual_block],
                dim= 1
                ).mean(dim= 1)
            
        predictions = self.postnet(decodings)

        return predictions


class Segment(torch.nn.Module):
    def forward(
        self,
        patterns: torch.Tensor,
        segment_size: int,
        lengths: torch.Tensor= None,
        offsets: torch.Tensor= None
        ):
        '''
        patterns: [Batch, Time, ...]
        lengths: [Batch]
        segment_size: an integer scalar    
        '''
        if offsets is None:
            offsets = (torch.rand_like(patterns[:, 0, 0]) * (lengths - segment_size)).long()
        segments = torch.stack([
            pattern[offset:offset + segment_size]
            for pattern, offset in zip(patterns, offsets)
            ], dim= 0)
        
        return segments, offsets

def Mask_Generate(lengths: torch.Tensor, max_length: Optional[Union[int, torch.Tensor]]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]


# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/transformer.py
class Positional_Encoding(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.demb = channels
        inv_freq = 1 / (10000 ** (torch.arange(0.0, channels, 2.0) / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        positional_sequence = torch.arange(x.size(2), device= x.device).to(x.dtype)
        sinusoid_inp = self.inv_freq.unsqueeze(1) @ positional_sequence.unsqueeze(0)
        positional_encodings = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim= 0)

        return x + positional_encodings.unsqueeze(0)