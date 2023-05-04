from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging
from typing import Dict, List, Optional
import functools

from Pattern_Generator import Text_Filtering, Phonemize

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

def Feature_Stack(features: List[np.ndarray], max_length: Optional[int]= None):
    max_feature_length = max_length or max([feature.shape[1] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, 0], [0, max_feature_length - feature.shape[1]]], constant_values= feature.min()) for feature in features],
        axis= 0
        )
    return features

def F0_Stack(f0s: List[np.ndarray], max_length: int= None):
    max_f0_length = max_length or max([f0.shape[0] for f0 in f0s])
    f0s = np.stack(
        [np.pad(f0, [0, max_f0_length - f0.shape[0]], constant_values= 0.0) for f0 in f0s],
        axis= 0
        )
    return f0s
def Audio_Stack(audios: List[np.ndarray], max_length: Optional[int]= None):
    max_audio_length = max_length or max([audio.shape[0] for audio in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]], constant_values= 0.0) for audio in audios],
        axis= 0
        )
    return audios

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        f0_info_dict: Dict[str, Dict[str, float]],
        pattern_path: str,
        metadata_file: str,
        feature_length_min: int,
        feature_length_max: int,
        text_length_min: int,
        text_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        self.token_dict = token_dict
        self.f0_info_dict = f0_info_dict
        self.pattern_path = pattern_path

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
                metadata_dict['Spectrogram_Length_Dict'][x] >= feature_length_min,
                metadata_dict['Spectrogram_Length_Dict'][x] <= feature_length_max,
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max
                ])
            ] * accumulated_dataset_epoch

        if use_pattern_cache:
            self.Pattern_LRU_Cache = functools.lru_cache(maxsize= None)(self.Pattern_LRU_Cache)
    
    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        return self.Pattern_LRU_Cache(path)
    
    def Pattern_LRU_Cache(self, path: str):
        pattern_dict = pickle.load(open(path, 'rb'))
        speaker = pattern_dict['Speaker']
        
        # padding between tokens
        token = ['<P>'] * (len(pattern_dict['Pronunciation']) * 2 - 1)
        token[0::2] = pattern_dict['Pronunciation']
        token = Text_to_Token(token, self.token_dict)

        f0 = pattern_dict['F0']        
        f0 = np.where(f0 != 0.0, (f0 - self.f0_info_dict[speaker]['Mean']) / self.f0_info_dict[speaker]['Std'], 0.0)

        return token, pattern_dict['GE2E'], pattern_dict['Spectrogram'], f0, pattern_dict['Audio']

    def __len__(self):
        return len(self.patterns)    

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        ge2e_dict: Dict[str, np.ndarray],
        texts: List[str],
        speakers: List[str],
        ):
        super().__init__()
        self.token_dict = token_dict
        self.ge2e_dict = ge2e_dict

        pronunciations = Phonemize(texts, language= 'English')

        self.patterns = []
        for index, (text, pronunciation, speaker) in enumerate(zip(texts, pronunciations, speakers)):
            text = Text_Filtering(text)
            if text is None or text == '':
                logging.warning('The text of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            elif speaker is None or not speaker in ge2e_dict.keys():
                logging.warning('The speaker of index {} is unknown. This index is ignoired.'.format(index))
                continue
            self.patterns.append((text, pronunciation, speaker))

    def __getitem__(self, idx):
        text, pronunciation, speaker = self.patterns[idx]

        token = ['<P>'] * (len(pronunciation) * 2 - 1)
        token[0::2] = pronunciation
        pronunciation = [(x if x != '<P>' else '') for x in token]
        token = Text_to_Token(token, self.token_dict)
        ge2e = self.ge2e_dict[speaker]

        return token, ge2e, text, pronunciation, speaker

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        hop_size: int,
        ):
        self.token_dict = token_dict
        self.hop_size = hop_size

    def __call__(self, batch):
        tokens, ge2es, features, f0s, audios  = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        feature_lengths = np.array([feature.shape[1] for feature in features])

        tokens = Token_Stack(
            tokens= tokens,
            token_dict= self.token_dict
            )
        ge2es = np.stack(ge2es, axis= 0)
        features = Feature_Stack(
            features= features
            )
        f0s = F0_Stack(
            f0s= f0s
            )
        audios = Audio_Stack(
            audios= audios
            )
        
        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        ge2es = torch.FloatTensor(ge2es)   # [Batch, GE2E_d]
        features = torch.FloatTensor(features)  # [Batch, Feature_d, Featpure_t]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]
        f0s = torch.FloatTensor(f0s)    # [Batch, Feature_t]
        audios = torch.FloatTensor(audios)    # [Batch, Audio_t], Audio_t == Feature_t * hop_size

        return tokens, token_lengths, ge2es, features, feature_lengths, f0s, audios

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, ge2es, texts, pronunciations, speakers = zip(*batch)

        token_lengths = np.array([token.shape[0] for token in tokens])
        
        tokens = Token_Stack(tokens, self.token_dict)
        ge2es = np.stack(ge2es, axis= 0)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        ge2es = torch.FloatTensor(ge2es)   # [Batch, GE2E_d]
        
        return tokens, token_lengths, ge2es, texts, pronunciations, speakers