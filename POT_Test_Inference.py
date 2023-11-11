import torch
import numpy as np
import logging, yaml, sys, os, time
from tqdm import tqdm
import tensorrt as trt
import onnxruntime
from random import sample
from unidecode import unidecode
from typing import Dict, List

from .Modules.Modules import VITS
from .Datasets import Inference_Dataset as Dataset, Token_Stack
from Arg_Parser import Recursive_Parse
from .Pattern_Generator import Text_Filtering

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

hp_path = './Exp3015/Hyper_Parameters.yaml'
checkpoint_path = './results/Exp3015/Checkpoint/S_2068.pt'
warm_up = 10

hp = Recursive_Parse(yaml.load(
    open(hp_path, encoding= 'utf-8'),
    Loader= yaml.Loader
    ))

class Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        fixed_length: int
        ):
        self.token_dict = token_dict
        self.fixed_length = fixed_length
         
    def __call__(self, batch):
        tokens, speakers, f0_means, f0_stds, *_ = zip(*batch)

        token_lengths = np.array([token.shape[0] for token in tokens])
        f0_means = np.array(f0_means)
        f0_stds = np.array(f0_stds)
        
        tokens = Token_Stack(
            tokens= tokens,
            token_dict= self.token_dict,
            max_length= self.fixed_length if type(self.fixed_length) == int else None
            )
        
        tokens = torch.IntTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.IntTensor(token_lengths)   # [Batch]
        speakers = torch.IntTensor(speakers)   # [Batch]
        f0_means = torch.FloatTensor(f0_means)    # [Batch]
        f0_stds = torch.FloatTensor(f0_stds)    # [Batch]
                
        return tokens, token_lengths, speakers, f0_means, f0_stds
    
def Generate_Test_Pattern(
    texts: List[str],
    speakers: List[str],
    languages: List[str],
    token_dict: Dict[str, int],
    speaker_dict: Dict[str, int],
    f0_info_dict: Dict[str, Dict[str, float]],
    fixed_length: int
    ):
    dataloader = torch.utils.data.DataLoader(
        dataset= Dataset(
            token_dict= token_dict,
            speaker_dict= speaker_dict,
            f0_info_dict= f0_info_dict,
            texts= texts,
            speakers= speakers,
            languages= languages
            ),
        shuffle= False,
        collate_fn= Collater(
            token_dict= token_dict,
            fixed_length= fixed_length
            ),
        batch_size= 1,
        num_workers= 0,
        pin_memory= True
        )
    
    dataloader.dataset.patterns = [
        (text, pronunciation, speaker)
        for text, pronunciation, speaker in dataloader.dataset.patterns
        if len(pronunciation) < 250
        ][:256]
    
    patterns = [
        (tokens, token_lengths, speakers, f0_means, f0_stds)
        for tokens, token_lengths, speakers, f0_means, f0_stds in dataloader
        ]
    
    return patterns

def Load_Engine(engine_filepath, trt_logger):
    with open(engine_filepath, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def _is_dimension_dynamic(dim):
    return dim is None or dim <= 0
def _is_shape_dynamic(shape):
    return any([_is_dimension_dynamic(dim) for dim in shape])
def Run_TRT_Engine(context, engine, tensors):
    bindings = [None]*engine.num_bindings
    for name,tensor in tensors['inputs'].items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()
        if engine.is_shape_binding(idx) and _is_shape_dynamic(context.get_shape(idx)):
            context.set_shape_input(idx, tensor)
        elif _is_shape_dynamic(engine.get_binding_shape(idx)):
            context.set_binding_shape(idx, tensor.shape)

    for name,tensor in tensors['outputs'].items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()

    context.execute_v2(bindings=bindings)

# pattern
token_dict = yaml.load(open(hp.Token_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
speaker_dict = yaml.load(open(hp.Speaker_Info_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
f0_info_dict = yaml.load(open(hp.F0_Info_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
patterns = []
for line in sample(open('D:/Rawdata/LJSpeech/metadata.csv', 'r', encoding= 'utf-8-sig').readlines(), 800):
    line = line.strip().split('|')        
    text = Text_Filtering(unidecode(line[2].strip()))
    if text is None:
        continue
    patterns.append((text, 'VCTK.S5', 'English'))
texts, speaker_labels, languages = zip(*patterns)

# model generate - pytorch
base_model = VITS(hp)
state_dict = torch.load(checkpoint_path, map_location= 'cpu')
base_model.load_state_dict(state_dict= state_dict['Model']['VITS'])
for parameters in base_model.parameters():
    parameters.requires_grad = False
base_model.decoder.Remove_Weight_Norm()
base_model.eval()

class Model(torch.nn.Module):
    def __init__(self, model, max_length):
        super().__init__()
        self.model = model
        self.max_length = max_length

    def forward(
        self,
        tokens: torch.Tensor,
        speakers: torch.Tensor,
        f0_means: torch.Tensor,
        f0_stds: torch.Tensor,
        ):
        encodings = self.model.text_encoder.token_embedding(tokens).permute(0, 2, 1)        
        for block in self.model.text_encoder.blocks:
            encodings = block.attention(
                queries= encodings,
                keys= encodings,
                values= encodings
                )
            encodings = block.ffn(encodings)

        encoding_means, encoding_stds = self.model.text_encoder.projection(encodings).chunk(chunks= 2, dim= 1)   # [Batch, Acoustic_d, Feature_t] * 2
        encoding_log_stds = torch.nn.functional.softplus(encoding_stds).log()

        speakers = self.model.speaker(speakers)

        encodings = (encodings + speakers.unsqueeze(2)).detach()
        durations = self.model.variance_block.duration_predictor(
            encodings= encodings
            )   # [Batch, Enc_t]
        
        repeats = (durations.float() + 0.5).int()
        feature_lengths = repeats.sum(dim=1)

        reps_cumsum = torch.cumsum(torch.nn.functional.pad(repeats, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]

        range_ = torch.arange(
            start= 0,
            end= self.max_length or feature_lengths.max(),
            step= 1,
            dtype= torch.int,
            device= encodings.device
            )[None, :, None]        
        alignments = ((reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)).float()

        f0s = self.model.variance_block.f0_predictor(
            encodings= encodings @ alignments.permute(0, 2, 1)
            )
        
        encoding_means = encoding_means @ alignments.permute(0, 2, 1)
        encoding_log_stds = encoding_log_stds @ alignments.permute(0, 2, 1)
        encoding_samples = encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds)
        
        acoustic_samples = self.model.acoustic_flow(
            x= encoding_samples,
            conditions= speakers,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]
        
        f0s = torch.where(
            condition= f0s.abs() < 0.001,
            input= f0s * f0_stds[:, None] + f0_means[:, None],
            other= torch.zeros_like(f0s)
            )
        f0_sines = self.model.decoder.hnnsf(
            x= f0s,
            upp= np.prod(hp.Decoder.Upsample.Rate)
            ).permute(0, 2, 1)  # [Batch, 1, Time]

        decodings = self.model.decoder.prenet(acoustic_samples)
        for upsample_block, noise_block, residual_blocks in zip(
            self.model.decoder.upsample_blocks,
            self.model.decoder.noise_blocks,
            self.model.decoder.residual_blocks,
            ):
            decodings = upsample_block(decodings) + noise_block(f0_sines)
            decodings = torch.stack(
                [block(decodings) for block in residual_blocks],
                # [block(decodings) for block in residual_block],
                dim= 1
                ).mean(dim= 1)
            
        predictions = self.model.decoder.postnet(decodings)

        return predictions

# open('./POT_Test/Exp3015.POT_Test.txt', 'a', encoding= 'utf-8-sig').write('Framework\tFixed_Length\tProcessor\tInput_Length\tTime\n')
for fixed_length, processor in [
    # ('Dynamic', 'CPU'),
    # ('Dynamic', 'GPU'),
    ]:
    model_torch = Model(base_model, None)
    model_torch.eval()

    patterns = Generate_Test_Pattern(
        texts= texts,
        speakers= speaker_labels,
        languages= languages,
        token_dict= token_dict,
        speaker_dict= speaker_dict,
        f0_info_dict= f0_info_dict,
        fixed_length= fixed_length // 2 if type(fixed_length) == int else None
        )
    tokens, _, speakers, f0_means, f0_stds = patterns[0]
    if processor == 'CPU':
        model_torch.cpu()
        device= 'cpu'
    else:
        model_torch.cuda()
        device= 'cuda:0'

    tokens = tokens.int().to(device)
    speakers = speakers.int().to(device)
    f0_means = f0_means.int().to(device)
    f0_stds = f0_stds.int().to(device)
    for _ in range(warm_up):
        with torch.inference_mode():
            predictions_torch = model_torch.forward(
                tokens= tokens,
                speakers= speakers,
                f0_means= f0_means,
                f0_stds= f0_stds
                )
            
    exports = []
    for tokens, token_lengths, speakers, f0_means, f0_stds in tqdm(
        patterns,
        desc= f'[PyTorch, {fixed_length}, {processor}]',
        total= len(patterns)
        ):
        tokens = tokens.int().to(device)
        speakers = speakers.int().to(device)
        f0_means = f0_means.int().to(device)
        f0_stds = f0_stds.int().to(device)
        with torch.inference_mode():
            st = time.time()
            predictions_torch = model_torch.forward(
                tokens= tokens,
                speakers= speakers,
                f0_means= f0_means,
                f0_stds= f0_stds
                )
            elapsed_time = (time.time() - st) * 1000
            exports.append(
                f'PyTorch\t{fixed_length}\t{processor}\t{token_lengths[0]}\t{elapsed_time}'
                )
    open('./POT_Test/Exp3015.POT_Test.txt', 'a', encoding= 'utf-8-sig').write('\n'.join(exports) + '\n')

for fixed_length, processor in [
    # ('Dynamic', 'CPU'),
    (1024, 'CPU'),
    (1024, 'GPU'),
    (2048, 'CPU'),
    (2048, 'GPU'),
    (4096, 'CPU'),
    (4096, 'GPU'),
    ]:
    model_onnx = onnxruntime.InferenceSession(
        f'./POT_Test/Exp3015.{fixed_length}.onnx',
        providers= ['CPUExecutionProvider' if processor == 'CPU' else 'CUDAExecutionProvider'],
        warmup_iterations= warm_up
        )
    
    patterns = Generate_Test_Pattern(
        texts= texts,
        speakers= speaker_labels,
        languages= languages,
        token_dict= token_dict,
        speaker_dict= speaker_dict,
        f0_info_dict= f0_info_dict,
        fixed_length= fixed_length // 2 if type(fixed_length) == int else None
        )
    
    exports = []
    for tokens, token_lengths, speakers, f0_means, f0_stds in tqdm(
        patterns,
        desc= f'[ONNX, {fixed_length}, {processor}]',
        total= len(patterns)
        ):
        tokens = tokens.int().numpy()
        speakers = speakers.int().numpy()
        f0_means = f0_means.numpy()
        f0_stds = f0_stds.numpy()

        st = time.time()
        predictions_onnx, = model_onnx.run(
            None,
            {
                'tokens': tokens,
                'speakers': speakers,
                'f0_means': f0_means,
                'f0_stds': f0_stds,
                }   
            )
        elapsed_time = (time.time() - st) * 1000
        exports.append(
            f'ONNX\t{fixed_length}\t{processor}\t{token_lengths[0]}\t{elapsed_time}'
            )
    open('./POT_Test/Exp3015.POT_Test.txt', 'a', encoding= 'utf-8-sig').write('\n'.join(exports) + '\n')

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
for fixed_length, processor in [
    (1024, 'GPU'),
    (2048, 'GPU'),
    (4096, 'GPU'),
    ]:
    model_trt_engine = Load_Engine(f'./POT_Test/Exp3015.{fixed_length}.trt', trt_logger= TRT_LOGGER)
    model_trt_context = model_trt_engine.create_execution_context()

    patterns = Generate_Test_Pattern(
        texts= texts,
        speakers= speaker_labels,
        languages= languages,
        token_dict= token_dict,
        speaker_dict= speaker_dict,
        f0_info_dict= f0_info_dict,
        fixed_length= fixed_length // 2 if type(fixed_length) == int else None
        )

    tokens, _, speakers, f0_means, f0_stds = patterns[0]

    predictions_trt = torch.zeros(1, fixed_length * hp.Sound.Frame_Shift, dtype= torch.float).cuda()
    for _ in range(warm_up):
        Run_TRT_Engine(
            context= model_trt_context,
            engine= model_trt_engine,
            tensors= {
                'inputs': {
                    'tokens': tokens.cuda(),
                    'speakers': speakers.cuda(),
                    'f0_means': f0_means.cuda(),
                    'f0_stds': f0_stds.cuda(),
                    },
                'outputs': {
                    'predictions': predictions_trt
                    }
                }
            )
    
    exports = []
    for tokens, token_lengths, speakers, f0_means, f0_stds in tqdm(
        patterns,
        desc= f'[ONNX, {fixed_length}, {processor}]',
        total= len(patterns)
        ):
        predictions_trt = torch.zeros(1, fixed_length * hp.Sound.Frame_Shift, dtype= torch.float).cuda()

        st = time.time()
        Run_TRT_Engine(
            context= model_trt_context,
            engine= model_trt_engine,
            tensors= {
                'inputs': {
                    'tokens': tokens.cuda(),
                    'speakers': speakers.cuda(),
                    'f0_means': f0_means.cuda(),
                    'f0_stds': f0_stds.cuda(),
                    },
                'outputs': {
                    'predictions': predictions_trt
                    }
                }
            )
        elapsed_time = (time.time() - st) * 1000
        exports.append(
            f'TRT\t{fixed_length}\t{processor}\t{token_lengths[0]}\t{elapsed_time}'
            )
    open('./POT_Test/Exp3015.POT_Test.txt', 'a', encoding= 'utf-8-sig').write('\n'.join(exports) + '\n')