from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging, librosa
from typing import Dict, List, Optional
import functools
from encodec import EncodecModel

from Pattern_Generator import Text_Filtering, Phonemize
from Modules.Nvidia_Alignment_Learning_Framework import Attention_Prior_Generator
     
def Text_to_Token(text: str, token_dict: Dict[str, int]):
    return np.array([
        token_dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Token_Stack(tokens: List[np.ndarray], token_dict, max_length: Optional[int]= None):
    max_token_length = max_length or max([token.shape[0] for token in tokens])
    tokens = np.stack(
        [np.pad(token, [0, max_token_length - token.shape[0]], constant_values= token_dict['<E>']) for token in tokens],
        axis= 0
        )
    return tokens

def Latent_Stack(latents: List[np.ndarray], max_length: Optional[int]= None):
    max_latent_length = max_length or max([latent.shape[1] for latent in latents])
    latents = np.stack(
        [np.pad(latent, [[0, 0], [0, max_latent_length - latent.shape[1]]], constant_values= 0) for latent in latents],
        axis= 0
        )
    return latents

def F0_Stack(f0s: List[np.ndarray], max_length: int= None):
    max_f0_length = max_length or max([f0.shape[0] for f0 in f0s])
    f0s = np.stack(
        [np.pad(f0, [0, max_f0_length - f0.shape[0]], constant_values= 0.0) for f0 in f0s],
        axis= 0
        )
    return f0s

def Mel_Stack(mels: List[np.ndarray], max_length: Optional[int]= None):
    max_mel_length = max_length or max([mel.shape[1] for mel in mels])
    mels = np.stack(
        [np.pad(mel, [[0, 0], [0, max_mel_length - mel.shape[1]]], constant_values= mel.min()) for mel in mels],
        axis= 0
        )
    return mels

def Attention_Prior_Stack(attention_priors: List[np.ndarray], max_token_length: int, max_latent_length: int):
    attention_priors_padded = np.zeros(
        shape= (len(attention_priors), max_latent_length, max_token_length),
        dtype= np.float32
        )    
    for index, attention_prior in enumerate(attention_priors):
        attention_priors_padded[index, :attention_prior.shape[0], :attention_prior.shape[1]] = attention_prior

    return attention_priors_padded

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        f0_info_dict: Dict[str, Dict[str, float]],
        use_between_padding: bool,
        pattern_path: str,
        metadata_file: str,
        latent_length_min: int,
        latent_length_max: int,
        text_length_min: int,
        text_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        self.token_dict = token_dict
        self.f0_info_dict = f0_info_dict
        self.use_between_padding = use_between_padding
        self.pattern_path = pattern_path

        self.attention_prior_generator = Attention_Prior_Generator()

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        
        self.patterns = []
        max_pattern_by_speaker = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Speaker_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Speaker_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_speaker)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns = [
            x for x in self.patterns
            if all([
                metadata_dict['Latent_Length_Dict'][x] >= latent_length_min,
                metadata_dict['Latent_Length_Dict'][x] <= latent_length_max,
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max
                ])
            ] * accumulated_dataset_epoch

        if use_pattern_cache:
            self.Pattern_LRU_Cache = functools.lru_cache(maxsize= None)(self.Pattern_LRU_Cache)
    
    def __getitem__(self, idx):
        '''
        compressed latent is for diffusion.
        non-compressed latent is for speech prompt.        
        '''
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        token, latent, f0, mel = self.Pattern_LRU_Cache(path)
        
        attention_prior = self.attention_prior_generator.get_prior(latent.shape[1], token.shape[0])

        return token, latent, f0, mel, attention_prior
    
    def Pattern_LRU_Cache(self, path: str):
        pattern_dict = pickle.load(open(path, 'rb'))
        speaker = pattern_dict['Speaker']
        
        if self.use_between_padding:
            # padding between tokens
            token = ['<P>'] * (len(pattern_dict['Pronunciation']) * 2 - 1)
            token[0::2] = pattern_dict['Pronunciation']
        else:
            token = pattern_dict['Pronunciation']
        token = Text_to_Token(token, self.token_dict)

        f0 = pattern_dict['F0']
        f0 = np.where(f0 != 0.0, (f0 - self.f0_info_dict[speaker]['Mean']) / self.f0_info_dict[speaker]['Std'], 0.0)
        f0 = np.clip(f0, -3.0, 3.0)

        return token, pattern_dict['Latent'], f0, pattern_dict['Mel']

    def __len__(self):
        return len(self.patterns)    

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        sample_rate: int,
        hop_size: int,
        use_between_padding: bool,
        texts: List[str],
        references: List[str]
        ):
        super().__init__()
        self.token_dict = token_dict
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.use_between_padding = use_between_padding
        self.encodec = EncodecModel.encodec_model_24khz()

        pronunciations = Phonemize(texts, language= 'English')

        self.patterns = []
        for index, (text, pronunciation, reference) in enumerate(zip(texts, pronunciations, references)):
            text = Text_Filtering(text)
            if text is None or text == '':
                logging.warning('The text of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            if not references is None and not os.path.exists(reference):
                logging.warning('The reference path of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            self.patterns.append((text, pronunciation, reference))

    def __getitem__(self, idx):
        text, pronunciation, reference = self.patterns[idx]

        if self.use_between_padding:
            token = ['<P>'] * (len(pronunciation) * 2 - 1)
            token[0::2] = pronunciation
            pronunciation = [(x if x != '<P>' else '') for x in token]
        else:
            token = pronunciation
        token = Text_to_Token(token, self.token_dict)

        audio, _ = librosa.load(reference, sr= self.sample_rate)
        audio = librosa.util.normalize(audio) * 0.95
        audio = audio[:audio.shape[0] - (audio.shape[0] % self.hop_size)]

        latent = self.encodec.encode(torch.from_numpy(audio)[None, None])[0][0].squeeze(0).short().numpy()  # [32, Audio_t / 320]

        return token, latent, text, pronunciation, reference

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens, latents, f0s, mels, attention_priors = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        latent_lengths = np.array([latent.shape[1] for latent in latents])
        speech_prompt_length = latent_lengths.min() // 2

        speech_prompts = []
        speech_prompts_for_diffusion = []
        for latent in latents:
            offset = np.random.randint(0, latent.shape[1] - speech_prompt_length + 1)
            speech_prompts.append(latent[:, offset:offset + speech_prompt_length])

            latent = np.concatenate([latent[:, 0:offset], latent[:, offset + speech_prompt_length:]], axis= 1)
            offset = np.random.randint(0, latent.shape[1] - speech_prompt_length + 1)
            speech_prompts_for_diffusion.append(latent[:, offset:offset + speech_prompt_length])

        tokens = Token_Stack(
            tokens= tokens,
            token_dict= self.token_dict
            )
        speech_prompts = Latent_Stack(speech_prompts)
        speech_prompts_for_diffusion = Latent_Stack(speech_prompts_for_diffusion)
        latents = Latent_Stack(
            latents= latents
            )
        f0s = F0_Stack(
            f0s= f0s
            )
        mels = Mel_Stack(
            mels= mels
            )
        attention_priors = Attention_Prior_Stack(
            attention_priors= attention_priors,
            max_token_length= token_lengths.max(),
            max_latent_length= latent_lengths.max()
            )
        
        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        speech_prompts = torch.LongTensor(speech_prompts)
        speech_prompts_for_diffusion = torch.LongTensor(speech_prompts_for_diffusion)
        latents = torch.LongTensor(latents)    # [Batch, Latent_d, Latent_t]
        latent_lengths = torch.LongTensor(latent_lengths)   # [Batch]
        f0s = torch.FloatTensor(f0s)    # [Batch, Latent_t]
        mels = torch.FloatTensor(mels)  # [Batch, Mel_d, Mel_t]
        attention_priors = torch.FloatTensor(attention_priors) # [Batch, Token_t, Latent_t]

        return tokens, token_lengths, speech_prompts, speech_prompts_for_diffusion, latents, latent_lengths, f0s, mels, attention_priors

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        speech_prompt_length: int
        ):
        self.token_dict = token_dict
        self.speech_prompt_length = speech_prompt_length
         
    def __call__(self, batch):
        tokens, speech_prompt_latents, texts, pronunciations, references = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        speech_prompt_latent_lengths = np.array([latent.shape[1] for latent in speech_prompt_latents])

        speech_prompt_length = min(self.speech_prompt_length, speech_prompt_latent_lengths.min())

        speech_prompts = []
        for latent in speech_prompt_latents:
            offset = np.random.randint(0, latent.shape[1] - speech_prompt_length + 1)
            speech_prompts.append(latent[:, offset:offset + speech_prompt_length])
        
        tokens = Token_Stack(tokens, self.token_dict)
        speech_prompts = Latent_Stack(speech_prompts)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        speech_prompts = torch.LongTensor(speech_prompts)
        
        return tokens, token_lengths, speech_prompts, texts, pronunciations, references