import torch
from typing import Optional
from .Layer import Conv1d, Linear, Lambda

# Daft-Exprt
class Gaussian_Upsampler(torch.nn.Module):
    def __init__(
        self,
        encoding_channels: int,   # encoidngs channels
        kernel_size: int,
        range_lstm_stack: int,
        range_dropout_rate: float,
        use_log_f0: bool= False,
        use_log_energy: bool= False,
        eps: float= 1e-5
        ):
        super().__init__()
        self.use_log_f0 = use_log_f0
        self.use_log_energy = use_log_energy

        self.duration_embedding = Conv1d(
            in_channels= 1,
            out_channels= encoding_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2
            )
        if use_log_f0:
            self.log_f0_embedding = Conv1d(
                in_channels= 1,
                out_channels= encoding_channels,
                kernel_size= kernel_size,
                padding= (kernel_size - 1) // 2
                )
        if use_log_energy:
            self.log_energy_embedding = Conv1d(
                in_channels= 1,
                out_channels= encoding_channels,
                kernel_size= kernel_size,
                padding= (kernel_size - 1) // 2
                )
        self.range_predictor = Range_Predictor(
            channels= encoding_channels,
            lstm_stack= range_lstm_stack,
            dropout_rate= range_dropout_rate
            )

        self.eps = eps

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor,
        durations: torch.Tensor,
        log_f0s: Optional[torch.Tensor]= None,
        log_energies: Optional[torch.Tensor]= None,
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        durations: [Batch, Enc_t]
        '''
        masks = Mask_Generate(lengths= encoding_lengths, max_length= torch.ones_like(durations[0]).sum())   # [Batch, Enc_t]

        encodings = encodings + self.duration_embedding(durations.unsqueeze(1).float())        
        if self.use_log_f0:
            encodings = encodings + self.log_f0_embedding(log_f0s.unsqueeze(1))
        if self.use_log_energy:
            encodings = encodings + self.log_energy_embedding(log_energies.unsqueeze(1))

        ranges = self.range_predictor(encodings, encoding_lengths)
        ranges.masked_fill_(masks, self.eps)

        duration_cumsums = durations.cumsum(dim= 1).float() - 0.5 * durations  # [Batch, Enc_t]
        duration_cumsums = duration_cumsums.unsqueeze(2)    # [Batch, Enc_t, 1]

        times = durations.sum(dim= 1, keepdim= True)   # [Batch, 1]
        times = torch.arange(1, times.max() + 1, device= durations.device)[None, None]  # [1, 1, Feature_t]

        ranges = ranges.unsqueeze(2)    # [Batch, Enc_t, 1]
        gaussians = torch.distributions.normal.Normal(loc= duration_cumsums, scale= ranges.clamp(self.eps))

        alignments = gaussians.log_prob(times).exp()    # [Batch, Enc_t, Feature_t]
        alignments = alignments.masked_fill(masks.unsqueeze(2), 0.0)
        alignments = alignments / (alignments.sum(dim= 1, keepdim= True) + self.eps)    # [Batch, Enc_t, Feature_t]
        
        return alignments


# Non-Attentive Tacotron
class Range_Predictor(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        lstm_stack: int,
        dropout_rate: float
        ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size= channels,
            hidden_size= channels // 2,
            num_layers= lstm_stack,
            bidirectional= True
            )
        self.lstm_dropout = torch.nn.Dropout(
            p= dropout_rate
            )

        self.projection = torch.nn.Sequential(
            Linear(
                in_features= channels,
                out_features= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.Softplus(),
            Lambda(lambda x: x.squeeze(2))
            )

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        '''
        masks = ~Mask_Generate(lengths= encoding_lengths, max_length= torch.ones_like(encodings[0, 0]).sum())   # [Batch, Enc_t]

        unpacked_length = encodings.size(2)
        encodings = encodings.permute(2, 0, 1)    # [Enc_t, Batch, Enc_d]        
        if self.training:
            encodings = torch.nn.utils.rnn.pack_padded_sequence(
                encodings,
                encoding_lengths.cpu().numpy(),
                enforce_sorted= False
                )
        
        self.lstm.flatten_parameters()
        encodings = self.lstm(encodings)[0]

        if self.training:
            encodings = torch.nn.utils.rnn.pad_packed_sequence(
                sequence= encodings,
                total_length= unpacked_length
                )[0]
        encodings = encodings.permute(1, 0, 2)    # [Batch, Enc_t, Enc_d]
        encodings = self.lstm_dropout(encodings)

        ranges = self.projection(encodings * masks.unsqueeze(2)) * masks  # [Batch, Enc_t]

        return ranges

def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]

