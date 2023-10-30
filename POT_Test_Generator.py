import torch
import numpy as np
import logging, yaml, sys, os, time, math
import tensorrt as trt
import onnxruntime
from scipy.io import wavfile
from typing import Dict

from Modules.Modules import NaturalSpeech2, Mask_Generate
from Datasets import Inference_Dataset as Dataset, Token_Stack, Latent_Stack
from Arg_Parser import Recursive_Parse

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

hp_path = './Hyper_Parameters.yaml'
checkpoint_path = './results/VCTK_230909/Checkpoint/S_85003.pt'
opset_version = 12
fixed_lengths = [1024]  # [1024, 2048, 4096]
os.makedirs('./POT_Test', exist_ok= True)

hp = Recursive_Parse(yaml.load(
    open(hp_path, encoding= 'utf-8'),
    Loader= yaml.Loader
    ))


class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        speech_prompt_length: int,
        fixed_length: int
        ):
        self.token_dict = token_dict
        self.speech_prompt_length = speech_prompt_length
        self.fixed_length = fixed_length
         
    def __call__(self, batch):
        tokens, latents, *_ = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        speech_prompt_lengths = np.array([latent.shape[1] for latent in latents])

        speech_prompt_length = min(self.speech_prompt_length, speech_prompt_lengths.min())
        speech_prompts = []
        for latent in latents:
            offset = np.random.randint(0, latent.shape[1] - speech_prompt_length + 1)
            speech_prompts.append(latent[:, offset:offset + speech_prompt_length])
        
        tokens = Token_Stack(
            tokens= tokens,
            token_dict= self.token_dict,
            max_length= self.fixed_length // 2 if type(self.fixed_length) == int else None
            )
        speech_prompts = Latent_Stack(
            latents= speech_prompts,
            max_length= self.speech_prompt_length
            )
        
        tokens = torch.IntTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.IntTensor(token_lengths)   # [Batch]
        speech_prompts = torch.IntTensor(speech_prompts)   # [Batch, Latent_t]        
                
        return tokens, token_lengths, speech_prompts
    
def Generate_Test_Pattern(
    text: str,
    reference: str,
    token_dict: Dict[str, int],
    fixed_length: int
    ):
    tokens, token_lengths, speech_prompts = list(iter(torch.utils.data.DataLoader(
        dataset= Dataset(
            token_dict= token_dict,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Frame_Shift,
            use_between_padding= hp.Use_Between_Padding,
            texts= [text],
            references= [reference],
            ),
        shuffle= False,
        collate_fn= Collater(
            token_dict= token_dict,
            speech_prompt_length= hp.Train.Inference_in_Train.Speech_Prompt_Length,
            fixed_length= fixed_length
            ),
        batch_size= 1,
        num_workers= 0,
        pin_memory= True
        )))[0]
    
    return tokens.int(), token_lengths.int(), speech_prompts.int()


def TRT_Build_Engine(model_file, shapes, max_ws= 512 * 1024 * 1024, fp16= False):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)

    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for s in shapes:
        profile.set_shape(s['name'], min=s['min'], opt=s['opt'], max=s['max'])
    config.add_optimization_profile(profile)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            for i in range(parser.num_errors):
                print("TensorRT ONNX parser error:", parser.get_error(i))
            engine = builder.build_engine(network, config=config)

            return engine

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

latent_info_dict = yaml.load(open(hp.Latent_Info_Path, 'r'), Loader=yaml.Loader)
latent_min = min([x['Min'] for x in latent_info_dict.values()])
latent_max = max([x['Max'] for x in latent_info_dict.values()])
base_model = NaturalSpeech2(
    hyper_parameters= hp,
    latent_min= latent_min,
    latent_max= latent_max
    )
state_dict = torch.load(checkpoint_path, map_location= 'cpu')
base_model.load_state_dict(state_dict= state_dict['Model'])
for parameters in base_model.parameters():
    parameters.requires_grad = False
base_model.eval()

class Encoder(torch.nn.Module):
    def __init__(
        self,
        model: NaturalSpeech2,
        max_length: int
        ):
        super().__init__()
        self.model = model
        self.max_length = max_length

    def forward(
        self,        
        tokens: torch.IntTensor,
        token_lengths: torch.IntTensor,
        speech_prompts: torch.IntTensor # type: ignore
        ):
        speech_prompts = self.model.hificodec.quantizer.embed(speech_prompts.permute(0, 2, 1)) # type: ignore
        encodings = self.model.encoder(
            tokens= tokens,
            lengths= token_lengths
            )
        speech_prompts: torch.FloatTensor = self.model.speech_prompter(speech_prompts)

        durations = self.model.variance_block.duration_predictor(
            encodings= encodings,
            speech_prompts= speech_prompts
            )   # [Batch, Enc_t]
        repeats = (durations.float() + 0.5).int()
        latent_lengths = repeats.sum(dim=1)
        reps_cumsum = torch.cumsum(torch.nn.functional.pad(repeats, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
        range_ = torch.arange(
            start= 0,
            end= self.max_length or latent_lengths.max(),
            step= 1,
            dtype= torch.int
            )[None, :, None]        
        alignments = ((reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)).float()
        encodings = encodings @ alignments.permute(0, 2, 1)

        f0s = self.model.variance_block.f0_predictor(
            encodings= encodings,
            speech_prompts= speech_prompts
            )   # [Batch, Latent_t]
        encodings = encodings + self.model.variance_block.f0_embedding(f0s.unsqueeze(1))

        encodings = self.model.frame_prior_network(
            encodings= encodings,
            lengths= latent_lengths
            )

        return encodings, speech_prompts, repeats
        # diffusion_predictions = self.model.diffusion.DDIM(
        #     encodings= encodings,
        #     lengths= latent_lengths,
        #     speech_prompts= speech_prompts,
        #     ddim_steps= 100
        #     )
        
        # diffusion_predictions = (diffusion_predictions.clamp(-1.0, 1.0) + 1.0) / 2.0 * (latent_max - latent_min) + latent_min

        # # Performing VQ to correct the incomplete predictions of diffusion.
        # *_, diffusion_predictions = self.model.hificodec.quantizer(diffusion_predictions)
        # diffusion_predictions = [code.reshape(tokens.size(0), -1) for code in diffusion_predictions]
        # diffusion_predictions = torch.stack(diffusion_predictions, 2)
        # diffusion_predictions = self.model.hificodec(diffusion_predictions).squeeze(1)
        
        # return diffusion_predictions

class Diffusion(torch.nn.Module):
    def __init__(
        self,
        model: NaturalSpeech2
        ):
        super().__init__()
        self.model = model
    
    def forward(
        self,        
        encodings: torch.FloatTensor,
        speech_prompts: torch.FloatTensor,
        features: torch.FloatTensor,
        diffusion_steps: torch.IntTensor,        
        ):
        diffusion_steps = torch.full(
            size= (encodings.size(0), ),
            fill_value= diffusion_steps,
            dtype= torch.int,
            device= encodings.device
            )
        
        # network를 풀어서 연산해볼것
        diffusion_step_embeddings = diffusion_steps.float() / hp.Diffusion.Max_Step
        starts = self.model.diffusion.network.prenet(features)

        encoding_residuals = encodings = self.model.diffusion.network.encoding_prenet(encodings)
        encodings = self.model.diffusion.network.encoding_ffn(encodings) # [Batch, Diffusion_d, Audio_ct]
        encodings = self.model.diffusion.network.encoding_norm(encodings + encoding_residuals)

        diffusion_step_embeddings = self.model.diffusion.network.step_embedding(diffusion_step_embeddings)
        diffusion_step_residuals = diffusion_step_embeddings[:, 1:, :]
        diffusion_step_embeddings = self.model.diffusion.network.step_ffn(diffusion_step_embeddings) # [Batch, Diffusion_d, 1]
        diffusion_step_embeddings = self.model.diffusion.network.step_norm(diffusion_step_embeddings + diffusion_step_residuals)

        speech_prompts = self.model.diffusion.network.pre_attention(
            query= self.model.diffusion.network.pre_attention_query.expand(-1, speech_prompts.size(0), -1),
            key= speech_prompts.permute(2, 0, 1),
            value= speech_prompts.permute(2, 0, 1)
            )[0].permute(1, 2, 0)   # [Batch, Diffusion_d, Token_n]

        skips_list = []
        for wavenet in self.model.diffusion.network.wavenets:
            starts, skips = wavenet(
                x= starts,
                masks= 1.0,                
                conditions= encodings,
                diffusion_steps= diffusion_step_embeddings,
                speech_prompts= speech_prompts
                )   # [Batch, Diffusion_d, Audio_ct]
            skips_list.append(skips)

        starts = torch.stack(skips_list, dim= 0).sum(dim= 0) / math.sqrt(hp.Diffusion.WaveNet.Stack)
        starts = self.model.diffusion.network.postnet(starts)

        # starts = self.model.diffusion.network(
        #     features= features,
        #     encodings= encodings,
        #     lengths= lengths,
        #     speech_prompts= speech_prompts,
        #     diffusion_steps= diffusion_steps
        #     )

        epsilons = \
            (features * self.model.diffusion.sqrt_recip_alphas_cumprod[diffusion_steps, None, None] - starts) / \
            self.model.diffusion.sqrt_recipm1_alphas_cumprod[diffusion_steps, None, None]
        
        # eta = 0.0, so sigmas and noises are zeros.
        direction_pointings = (1.0 - self.model.diffusion.alphas_cumprod_prev[diffusion_steps, None, None]) * epsilons
        noises = 0.0

        features = self.model.diffusion.alphas_cumprod_prev[diffusion_steps, None, None].sqrt() * starts + direction_pointings + noises

        return features

class Postnet(torch.nn.Module):
    def __init__(
        self,
        model: NaturalSpeech2,
        latent_min: float,
        latent_max: float
        ):
        super().__init__()
        self.model = model
        self.latent_min = latent_min
        self.latent_max = latent_max

    def forward(
        self,        
        features: torch.FloatTensor
        ):
        predictions = (features.clamp(-1.0, 1.0) + 1.0) / 2.0 * (self.latent_max - self.latent_min) + self.latent_min

        *_, predictions = self.model.hificodec.quantizer(predictions)
        predictions = torch.stack(predictions, dim= 1)[None]
        predictions = self.model.hificodec(predictions).squeeze(1)
        
        return predictions


for fixed_length in fixed_lengths:
    logging.info(f'Fixed length: {fixed_length}')
    tokens, token_lengths, speech_prompts = Generate_Test_Pattern(
        text= 'Do not kill the goose that lays the golden eggs.',
        reference= './Inference_Wav/s5_004_mic2.flac',
        token_dict= yaml.load(open(hp.Token_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader),
        fixed_length= fixed_length
        )
    
    logging.info(f'Token size: {tokens.size()}')
    logging.info(f'Token length size: {token_lengths.size()}')
    logging.info(f'Speech prompts size: {speech_prompts.size()}')

    # model - pytorch
    logging.info(f'Pytorch - Model generate')
    encoder_torch = Encoder(base_model, fixed_length)
    diffusion_torch = Diffusion(base_model)
    postnet_torch = Postnet(
        model= base_model,
        latent_min= latent_min,
        latent_max= latent_max
        )

    # pytorch tensor
    logging.info(f'Pytorch - Inference test')
    with torch.inference_mode():
        encodings_torch, speech_encodings_torch, durations_torch = encoder_torch.forward(
            tokens= tokens,
            token_lengths= token_lengths,
            speech_prompts= speech_prompts
            )
        latent_lengths_torch = durations_torch[:, :token_lengths[0] - 1].sum(dim= 1)

        ddim_steps = torch.arange(0, hp.Diffusion.Max_Step, hp.Diffusion.Max_Step // 100).int().flip(dims= [0])
        features_torch = torch.randn(
            size= (encodings_torch.size(0), hp.Audio_Codec_Size, encodings_torch.size(2)),
            dtype= encodings_torch.dtype,
            device= encodings_torch.device
            )
        for diffusion_steps in ddim_steps:
            features_torch = diffusion_torch.forward(
                encodings= encodings_torch,
                speech_prompts= speech_encodings_torch,
                features= features_torch,
                diffusion_steps= diffusion_steps
                )
            
        predictions_torch = postnet_torch.forward(features_torch)
        predictions_torch = predictions_torch[:, :(latent_lengths_torch[0] - 1) * hp.Sound.Frame_Shift]


    # pytorch-to-onnx
    logging.info(f'ONNX - Export')
    torch.onnx.export(
        model= encoder_torch,
        args= (tokens, token_lengths, speech_prompts),
        f= f'./POT_Test/{fixed_length}.Encoder.onnx',
        opset_version= opset_version,
        do_constant_folding= True,
        input_names= ['tokens', 'token_lengths', 'speech_prompts'],
        output_names= ['encodings', 'speech_encodings', 'durations']
        )
    torch.onnx.export(
        model= diffusion_torch,
        args= (encodings_torch, speech_encodings_torch, features_torch, diffusion_steps),
        f= f'./POT_Test/{fixed_length}.Diffusion.onnx',
        opset_version= opset_version,
        do_constant_folding= True,
        input_names= ['encodings', 'speech_encodings', 'feature_inputs', 'diffusion_steps'],
        output_names= ['features',]
        )
    torch.onnx.export(
        model= postnet_torch,
        args= (features_torch,),
        f= f'./POT_Test/{fixed_length}.Postnet.onnx',
        opset_version= opset_version,
        do_constant_folding= True,
        input_names= ['features'],
        output_names= ['predictions',]
        )
    
    # model - onnx
    logging.info(f'ONNX - Model generate')
    encoder_onnx = onnxruntime.InferenceSession(
        f'./POT_Test/{fixed_length}.Encoder.onnx',
        providers= ['CPUExecutionProvider']
        )
    diffusion_onnx = onnxruntime.InferenceSession(
        f'./POT_Test/{fixed_length}.Diffusion.onnx',
        providers= ['CPUExecutionProvider']
        )
    postnet_onnx = onnxruntime.InferenceSession(
        f'./POT_Test/{fixed_length}.Postnet.onnx',
        providers= ['CPUExecutionProvider']
        )
    
    # onnx tensor
    logging.info(f'ONNX - Inference')
    encodings_onnx, speech_encodings_onnx, durations_onnx = encoder_onnx.run(
        None,
        {
            'tokens': tokens.cpu().numpy(),
            'token_lengths': token_lengths.cpu().numpy(),
            'speech_prompts': speech_prompts.cpu().numpy(),
            }   
        )
    latent_lengths_onnx = durations_onnx[:, :token_lengths[0] - 1].sum(axis= 1)
    ddim_steps = torch.arange(0, hp.Diffusion.Max_Step, hp.Diffusion.Max_Step // 100).int().flip(dims= [0]).cpu().numpy()
    features_onnx = np.random.randn(encodings_onnx.shape[0], hp.Audio_Codec_Size, encodings_onnx.shape[2]).astype(np.float32)
    for diffusion_steps in ddim_steps:
        features_onnx, = diffusion_onnx.run(
            None,
            {
                'encodings': encodings_onnx,
                'speech_encodings': speech_encodings_onnx,
                'feature_inputs': features_onnx,
                'diffusion_steps': diffusion_steps[None]
                }
            )
    predictions_onnx, = postnet_onnx.run(
            None,
            {
                'features': features_onnx,
                }
            )
            
    predictions_onnx = predictions_onnx[:, :(latent_lengths_onnx[0] - 1) * hp.Sound.Frame_Shift]
    predictions_onnx = torch.from_numpy(predictions_onnx)
    
    # onnx-to-trt
    logging.info(f'TensorRT - Export')
    encoder_shapes = [
        {
            'name': 'tokens',
            'min': (1, fixed_length // 2),
            'opt': (1, fixed_length // 2),
            'max': (1, fixed_length // 2)
            },
        {
            'name': 'token_lengths',
            'min': (1,),
            'opt': (1,),
            'max': (1,)
            },
        {
            'name': 'speech_prompts',
            'min': (1, 4, hp.Train.Inference_in_Train.Speech_Prompt_Length),
            'opt': (1, 4, hp.Train.Inference_in_Train.Speech_Prompt_Length),
            'max': (1, 4, hp.Train.Inference_in_Train.Speech_Prompt_Length)
            },
        ]
    diffusion_shapes = [
        {
            'name': 'encodings',
            'min': (1, hp.Encoder.Size, fixed_length),
            'opt': (1, hp.Encoder.Size, fixed_length),
            'max': (1, hp.Encoder.Size, fixed_length)
            },
        {
            'name': 'speech_encodings',
            'min': (1, hp.Speech_Prompter.Size, hp.Train.Inference_in_Train.Speech_Prompt_Length),
            'opt': (1, hp.Speech_Prompter.Size, hp.Train.Inference_in_Train.Speech_Prompt_Length),
            'max': (1, hp.Speech_Prompter.Size, hp.Train.Inference_in_Train.Speech_Prompt_Length)
            },
        {
            'name': 'features',
            'min': (1, hp.Audio_Codec_Size, fixed_length),
            'opt': (1, hp.Audio_Codec_Size, fixed_length),
            'max': (1, hp.Audio_Codec_Size, fixed_length)
            },
        {
            'name': 'diffusion_steps',
            'min': (),
            'opt': (),
            'max': ()
            },
        ]
    postnet_shapes = [
        {
            'name': 'features',
            'min': (1, hp.Audio_Codec_Size, fixed_length),
            'opt': (1, hp.Audio_Codec_Size, fixed_length),
            'max': (1, hp.Audio_Codec_Size, fixed_length)
            },
        ]
    encoder_trt_engine = TRT_Build_Engine(
        model_file= f'./POT_Test/{fixed_length}.Encoder.onnx',
        shapes= encoder_shapes,
        max_ws= 4 * 1024 ** 3,
        fp16= True
        )
    diffusion_trt_engine = TRT_Build_Engine(
        model_file= f'./POT_Test/{fixed_length}.Diffusion.onnx',
        shapes= diffusion_shapes,
        max_ws= 4 * 1024 ** 3,
        fp16= True
        )
    postnet_trt_engine = TRT_Build_Engine(
        model_file= f'./POT_Test/{fixed_length}.Postnet.onnx',
        shapes= postnet_shapes,
        max_ws= 4 * 1024 ** 3,
        fp16= True
        )
    with open(f'./POT_Test/{fixed_length}.Encoder.trt', 'wb') as f:
        f.write(encoder_trt_engine.serialize())
    with open(f'./POT_Test/{fixed_length}.Diffusion.trt', 'wb') as f:
        f.write(diffusion_trt_engine.serialize())
    with open(f'./POT_Test/{fixed_length}.Postnet.trt', 'wb') as f:
        f.write(postnet_trt_engine.serialize())

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # model - trt
    logging.info(f'TensorRT - Model generate')
    encoder_trt_engine = Load_Engine(f'./POT_Test/{fixed_length}.Encoder.trt', trt_logger= TRT_LOGGER)
    encoder_trt_context = encoder_trt_engine.create_execution_context()
    diffusion_trt_engine = Load_Engine(f'./POT_Test/{fixed_length}.Diffusion.trt', trt_logger= TRT_LOGGER)
    diffusion_trt_context = diffusion_trt_engine.create_execution_context()
    postnet_trt_engine = Load_Engine(f'./POT_Test/{fixed_length}.Postnet.trt', trt_logger= TRT_LOGGER)
    postnet_trt_context = postnet_trt_engine.create_execution_context()

    # trt tensor
    logging.info(f'TensorRT - Inference')
    encodings_trt = torch.zeros(1, fixed_length * hp.Sound.Frame_Shift, dtype= torch.float).cuda()
    speech_encodings_trt = torch.zeros(1, hp.Speech_Prompter.Size, hp.Train.Inference_in_Train.Speech_Prompt_Length, dtype= torch.float).cuda()
    durations_trt = torch.zeros(1, fixed_length // 2, dtype= torch.int).cuda()    
    Run_TRT_Engine(
        context= encoder_trt_context,
        engine= encoder_trt_engine,
        tensors= {
            'inputs': {
                'tokens': tokens.cuda(),
                'token_lengths': token_lengths.cuda(),
                'speech_prompts': speech_prompts.cuda(),
                },
            'outputs': {
                'encodings': encodings_trt,
                'speech_encodings': speech_encodings_trt,
                'durations': durations_trt
                }
            }
        )
    latent_lengths_trt = durations_trt[:, :token_lengths[0] - 1].sum(dim= 1)
    ddim_steps = torch.arange(0, hp.Diffusion.Max_Step, hp.Diffusion.Max_Step // 100).int().flip(dims= [0])
    feature_inputs_trt = torch.randn(1, hp.Audio_Codec_Size, fixed_length, dtype= torch.float).cuda()
    for diffusion_steps in ddim_steps:
        features_trt = torch.zeros(1, hp.Audio_Codec_Size, fixed_length, dtype= torch.float).cuda()
        Run_TRT_Engine(
            context= diffusion_trt_context,
            engine= diffusion_trt_engine,
            tensors= {
                'inputs': {
                    'encodings': encodings_trt.cuda(),
                    'speech_encodings': speech_encodings_trt.cuda(),
                    'feature_inputs': feature_inputs_trt.cuda(),
                    'diffusion_steps': diffusion_steps.cuda(),
                    },
                'outputs': {
                    'features': features_trt,
                    }
                }
            )
        feature_inputs_trt = features_trt

    predictions_trt = torch.zeros(1, fixed_length * hp.Sound.Frame_Shift, dtype= torch.float).cuda()
    Run_TRT_Engine(
        context= postnet_trt_context,
        engine= postnet_trt_engine,
        tensors= {
            'inputs': {
                'features': features_trt.cuda(),
                },
            'outputs': {
                'predictions': predictions_trt,
                }
            }
        )
    predictions_trt = predictions_trt[:, :(latent_lengths_trt[0] - 1) * hp.Sound.Frame_Shift].cpu()
    
# dynamic
logging.info(f'Fixed length: Dynamic')
tokens, token_lengths, speech_prompts = Generate_Test_Pattern(
    text= 'Do not kill the goose that lays the golden eggs.',
    reference= './Inference_Wav/s5_004_mic2.flac',
    token_dict= yaml.load(open(hp.Token_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader),
    fixed_length= None
    )

logging.info(f'Pytorch - Model generate')
encoder_dynamic_torch = Encoder(base_model, None)
diffusion_dynamic_torch = Diffusion(base_model)
postnet_dynamic_torch = Postnet(
    model= base_model,
    latent_min= latent_min,
    latent_max= latent_max
    )
logging.info(f'Pytorch - Inference test')
with torch.inference_mode():
    encodings_dynamic_torch, speech_encodings_dynamic_torch, durations_dynamic_torch = encoder_dynamic_torch.forward(
        tokens= tokens,
        token_lengths= token_lengths,
        speech_prompts= speech_prompts
        )

    ddim_steps = torch.arange(0, hp.Diffusion.Max_Step, hp.Diffusion.Max_Step // 100).int().flip(dims= [0])
    features_dynamic_torch = torch.randn(
        size= (encodings_dynamic_torch.size(0), hp.Audio_Codec_Size, encodings_dynamic_torch.size(2)),
        dtype= encodings_dynamic_torch.dtype,
        device= encodings_dynamic_torch.device
        )
    for diffusion_steps in ddim_steps:
        features_dynamic_torch = diffusion_dynamic_torch.forward(
            encodings= encodings_dynamic_torch,
            speech_prompts= speech_encodings_dynamic_torch,
            features= features_dynamic_torch,
            diffusion_steps= diffusion_steps
            )        
    predictions_dynamic_torch = postnet_dynamic_torch.forward(features_dynamic_torch)

# pytorch-to-onnx
logging.info(f'ONNX - Export')
torch.onnx.export(
    model= encoder_dynamic_torch,
    args= (tokens, token_lengths, speech_prompts),
    f= f'./POT_Test/Dynamic.Encoder.onnx',
    opset_version= opset_version,
    do_constant_folding= True,
    input_names= ['tokens', 'token_lengths', 'speech_prompts'],
    output_names= ['encodings', 'speech_encodings', 'durations'],
    dynamic_axes= {
        'tokens': {1: 'token_length'},
        'encodings': {2: 'latent_length'},
        'durations': {1: 'token_length'}
        }
    )
torch.onnx.export(
    model= diffusion_dynamic_torch,
    args= (encodings_dynamic_torch, speech_encodings_dynamic_torch, features_dynamic_torch, diffusion_steps),
    f= f'./POT_Test/Dynamic.Diffusion.onnx',
    opset_version= opset_version,
    do_constant_folding= True,
    input_names= ['encodings', 'speech_encodings', 'feature_inputs', 'diffusion_steps'],
    output_names= ['features',],
    dynamic_axes= {
        'encodings': {2: 'latent_length'},
        'feature_inputs': {2: 'latent_length'},
        'features': {2: 'latent_length'}
        }
    )
torch.onnx.export(
    model= postnet_dynamic_torch,
    args= (features_dynamic_torch,),
    f= f'./POT_Test/Dynamic.Postnet.onnx',
    opset_version= opset_version,
    do_constant_folding= True,
    input_names= ['features'],
    output_names= ['predictions'],
    dynamic_axes= {
        'features': {2: 'latent_length'},
        'predictions': {1: 'audio_length'}
        }
    )

# model - onnx
logging.info(f'ONNX - Model generate')
encoder_dynamic_onnx = onnxruntime.InferenceSession(
    f'./POT_Test/Dynamic.Encoder.onnx',
    providers= ['CPUExecutionProvider']
    )
diffusion_dynamic_onnx = onnxruntime.InferenceSession(
    f'./POT_Test/Dynamic.Diffusion.onnx',
    providers= ['CPUExecutionProvider']
    )
postnet_dynamic_onnx = onnxruntime.InferenceSession(
    f'./POT_Test/Dynamic.Postnet.onnx',
    providers= ['CPUExecutionProvider']
    )

# onnx tensor
logging.info(f'ONNX - Inference')
encodings_dynamic_onnx, speech_encodings_dynamic_onnx, durations_dynamic_onnx = encoder_dynamic_onnx.run(
    None,
    {
        'tokens': tokens.cpu().numpy(),
        'token_lengths': token_lengths.cpu().numpy(),
        'speech_prompts': speech_prompts.cpu().numpy(),
        }
    )
ddim_steps = torch.arange(0, hp.Diffusion.Max_Step, hp.Diffusion.Max_Step // 100).int().flip(dims= [0]).cpu().numpy()
features_dynamic_onnx = np.random.randn(encodings_dynamic_onnx.shape[0], hp.Audio_Codec_Size, encodings_dynamic_onnx.shape[2]).astype(np.float32)
for diffusion_steps in ddim_steps:
    features_dynamic_onnx, = diffusion_dynamic_onnx.run(
        None,
        {
            'encodings': encodings_dynamic_onnx,
            'speech_encodings': speech_encodings_dynamic_onnx,
            'feature_inputs': features_dynamic_onnx,
            'diffusion_steps': diffusion_steps[None]
            }
        )
predictions_dynamic_onnx, = postnet_dynamic_onnx.run(
        None,
        {
            'features': features_dynamic_onnx,
            }
        )
        
predictions_dynamic_onnx = torch.from_numpy(predictions_dynamic_onnx)