import torch
import numpy as np
import yaml, os, pickle, librosa, re, argparse, math
from concurrent.futures import ThreadPoolExecutor as PE
from random import shuffle
from tqdm import tqdm
from pysptk.sptk import rapt
from typing import List, Tuple, Dict, Union, Optional

from phonemizer import phonemize
from unidecode import unidecode

from meldataset import mel_spectrogram

from hificodec.vqvae import VQVAE

from Arg_Parser import Recursive_Parse

using_extension = [x.upper() for x in ['.wav', '.m4a', '.flac', '.flac']]
regex_checker = re.compile('[가-힣A-Za-z,.?!\'\-\s]+')

if __name__ == '__main__':
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    hificodec = VQVAE(
        config_path= './hificodec/config_24k_320d.json',
        ckpt_path= './hificodec/HiFi-Codec-24k-320d',
        with_encoder= True
        ).to(device)

def Text_Filtering(text: str):
    remove_letter_list = ['(', ')', '\"', '[', ']', ':', ';']
    replace_list = [('  ', ' '), (' ,', ','), ('\' ', '\''), ('“', ''), ('”', ''), ('’', '\'')]

    text = text.strip()
    for filter in remove_letter_list:
        text= text.replace(filter, '')
    for filter, replace_STR in replace_list:
        text= text.replace(filter, replace_STR)

    text= text.strip()
    
    if len(regex_checker.findall(text)) != 1:
        return None
    elif text.startswith('\''):
        return None
    else:
        return regex_checker.findall(text)[0]

whitespace_re = re.compile(r'\s+')
english_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ]]
def expand_abbreviations(text: str):
    for regex, replacement in english_abbreviations:
        text = re.sub(regex, replacement, text)

    return text

def Phonemize(texts: Union[str, List[str]], language: str):
    if type(texts) == str:
        texts = [texts]

    if language == 'English':
        language = 'en-us'
        # English cleaners 2
        texts = [text.lower() for text in texts]
        texts = [expand_abbreviations(text) for text in texts]
    elif language == 'Korean':
        language = 'ko'

    pronunciations = phonemize(
        texts,
        language= language,
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress= True,
        njobs=4
        )
    pronunciations = [re.sub(whitespace_re, ' ', pronunciation) for pronunciation in pronunciations]

    return pronunciations

def Pattern_Generate(
    path,
    sample_rate: int,
    hop_size: int,
    num_mels: int,
    f0_min: int,
    f0_max: int
    ):
    audio, _ = librosa.load(path, sr= sample_rate)
    audio = librosa.util.normalize(audio) * 0.95
    audio = audio[:audio.shape[0] - (audio.shape[0] % hop_size)]
    audio_tensor = torch.from_numpy(audio).to(device).float()[None]
    with torch.inference_mode():
        latent = hificodec.encode(audio_tensor)[0].T.cpu().numpy() # [4, Audio_t / 320]
        mel = mel_spectrogram(
            y= audio_tensor,
            n_fft= hop_size * 4,
            num_mels= num_mels,
            sampling_rate= sample_rate,
            hop_size= hop_size,
            win_size= hop_size * 4,
            fmin= 0,
            fmax= None,
            center= False
            )[0].cpu().numpy()

    f0 = rapt(
        x= audio * 32768,
        fs= sample_rate,
        hopsize= hop_size,
        min= f0_min,
        max= f0_max,
        otype= 1
        )
    
    if abs(latent.shape[1] - f0.shape[0]) > 1:
        return None, None, None, None
    elif latent.shape[1] > f0.shape[0]:
        f0 = np.pad(f0, [0, latent.shape[1] - f0.shape[0]], constant_values= 0.0)
    else:   # mel.shape[1] < f0.shape[0]:
        audio = np.pad(audio, [0, (f0.shape[0] - latent.shape[1]) * hop_size])
        latent = np.pad(latent, [[0, 0], [0, f0.shape[0] - latent.shape[1]]], mode= 'edge')

    if mel.shape[1] - f0.shape[0] < 0 or mel.shape[1] - f0.shape[0] > 1:
        return None, None, None, None
    else:
        mel = mel[:, :f0.shape[0]]
        
    nonsilence_frames = np.where(f0 > 0.0)[0]
    if len(nonsilence_frames) < 2:
        return None, None, None, None
    initial_silence_frame, *_, last_silence_frame = nonsilence_frames
    initial_silence_frame = max(initial_silence_frame - 11, 0)
    last_silence_frame = min(last_silence_frame + 11, f0.shape[0])
    audio = audio[initial_silence_frame * hop_size:last_silence_frame * hop_size]
    latent = latent[:, initial_silence_frame:last_silence_frame]
    mel = mel[:, initial_silence_frame:last_silence_frame]
    f0 = f0[initial_silence_frame:last_silence_frame]
    
    return audio.astype(np.float16), latent.astype(np.int16), mel.astype(np.float16), f0.astype(np.float16)

def Pattern_File_Generate(path: str, speaker: str, emotion: str, language: str, gender: str, dataset: str, text: str, pronunciation: str, tag: str='', eval: bool= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path

    file = '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()
    if any([
        os.path.exists(os.path.join(x, dataset, speaker, file).replace("\\", "/"))
        for x in [hp.Train.Eval_Pattern.Path, hp.Train.Train_Pattern.Path]
        ]):
        return
    file = os.path.join(pattern_path, dataset, speaker, file).replace("\\", "/")

    audio, latent, mel, f0 = Pattern_Generate(
        path= path,
        sample_rate= hp.Sound.Sample_Rate,
        hop_size= hp.Sound.Frame_Shift,
        num_mels= hp.Sound.Mel_Dim,
        f0_min= hp.Sound.F0_Min,
        f0_max= hp.Sound.F0_Max
        )
    if audio is None:
        return
     
    new_Pattern_dict = {
        'Audio': audio,
        'Latent': latent,
        'Mel': mel,
        'F0': f0,
        'Speaker': speaker,
        'Emotion': emotion,
        'Language': language,
        'Gender': gender,
        'Dataset': dataset,
        'Text': text,
        'Pronunciation': pronunciation
        }

    os.makedirs(os.path.join(pattern_path, dataset, speaker).replace('\\', '/'), exist_ok= True)
    with open(file, 'wb') as f:
        pickle.dump(new_Pattern_dict, f, protocol=4)

def Selvas_Info_Load(path: str):
    '''
    ema, emb, emf, emg, emh, nea, neb, nec, ned, nee, nek, nel, nem, nen, neo
    1-100: Neutral
    101-200: Happy
    201-300: Sad
    301-400: Angry

    lmy, ava, avb, avc, avd, ada, adb, adc, add:
    all neutral
    '''
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            if any(['lmy04282' in file, 'lmy07365' in file, 'lmy05124' in file]):
                continue
            paths.append(file)

    text_dict = {}
    
    for wav_path in paths:
        text = open(wav_path.replace('/wav/', '/transcript/').replace('.wav', '.txt'), 'r', encoding= 'utf-8-sig').readlines()[0].strip()
        text = Text_Filtering(text)
        if text is None:
            continue

        text_dict[wav_path] = text

    paths = list(text_dict.keys())

    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'Korean'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        path: path.split('/')[-3].strip().upper()
        for path in paths
        }
    
    emotion_dict = {}
    for path in paths:
        if speaker_dict[path] in ['LMY', 'KIH', 'AVA', 'AVB', 'AVC', 'AVD', 'ADA', 'ADB', 'ADC', 'ADD', 'PFA', 'PFB', 'PFC', 'PFD', 'PFI', 'PFL', 'PFM', 'PFO', 'PFP', 'PMA', 'PMB', 'PMC', 'PMD', 'PMI', 'PMJ', 'PML']:
            emotion_dict[path] = 'Neutral'
        elif speaker_dict[path] in ['EMA', 'EMB', 'EMF', 'EMG', 'EMH', 'NEA', 'NEB', 'NEC', 'NED', 'NEE', 'NEK', 'NEL', 'NEM', 'NEN', 'NEO']:
            index = int(os.path.splitext(os.path.basename(path))[0][-5:])
            if index > 0 and index < 101:
                emotion_dict[path] = 'Neutral'
            elif index > 100 and index < 201:
                emotion_dict[path] = 'Happy'
            elif index > 200 and index < 301:
                emotion_dict[path] = 'Sad'
            elif index > 300 and index < 401:
                emotion_dict[path] = 'Angry'
            else:
                raise NotImplementedError('Unknown emotion index: {}'.format(index))
        else:
            raise NotImplementedError('Unknown speaker: {}'.format(speaker_dict[path]))

    language_dict = {path: 'Korean' for path in paths}

    gender_dict = {
        'ADA': 'Female',
        'ADB': 'Female',
        'ADC': 'Male',
        'ADD': 'Male',
        'AVA': 'Female',
        'AVB': 'Female',
        'AVC': 'Female',
        'AVD': 'Female',
        'EMA': 'Female',
        'EMB': 'Female',
        'EMF': 'Male',
        'EMG': 'Male',
        'EMH': 'Male',
        'KIH': 'Male',
        'LMY': 'Female',
        'NEA': 'Female',
        'NEB': 'Female',
        'NEC': 'Female',
        'NED': 'Female',
        'NEE': 'Female',
        'NEK': 'Male',
        'NEL': 'Male',
        'NEM': 'Male',
        'NEN': 'Male',
        'NEO': 'Male',
        'PFA': 'Female',
        'PFB': 'Female',
        'PFC': 'Female',
        'PFD': 'Female',
        'PFI': 'Female',
        'PFL': 'Female',
        'PFM': 'Female',
        'PFO': 'Female',
        'PFP': 'Female',
        'PMA': 'Male',
        'PMB': 'Male',
        'PMC': 'Male',
        'PMD': 'Male',
        'PMI': 'Male',
        'PMJ': 'Male',
        'PML': 'Male',
        }
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print('Selvas info generated: {}'.format(len(paths)))
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def KSS_Info_Load(path: str):
    '''
    all neutral
    '''
    paths, text_dict = [], {}
    for line in open(os.path.join(path, 'transcript.v.1.4.txt').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines():
        line = line.strip().split('|')
        file, text = line[0].strip(), line[2].strip()
        text = Text_Filtering(text)
        if text is None:
            continue
        
        file = os.path.join(path, 'kss', file).replace('\\', '/')
        paths.append(file)
        text_dict[file] = text

    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'Korean'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        path: 'KSS'
        for path in paths
        }
    emotion_dict = {
        path: 'Neutral'
        for path in paths
        }
    language_dict = {
        path: 'Korean'
        for path in paths
        }
    gender_dict = {
        path: 'Female'
        for path in paths
        }

    print('KSS info generated: {}'.format(len(paths)))
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def AIHub_Info_Load(path: str):
    emotion_label_dict = {
        'Neutrality': 'Neutral'
        }

    skip_info_keys = []
    info_dict = {}
    for root, _, files in os.walk(path):
        for file in files:
            key, extension = os.path.splitext(file)
            if not key in info_dict.keys():
                info_dict[key] = {}
            
            path = os.path.join(root, file).replace('\\', '/')
            if extension.upper() == '.WAV':
                info_dict[key]['Path'] = path
            elif extension.upper() == '.JSON':
                pattern_info = json.load(open(path, encoding= 'utf-8-sig'))
                text = Text_Filtering(pattern_info['전사정보']['TransLabelText'].replace('\xa0', ' '))
                if text is None:
                    skip_info_keys.append(key)
                    continue

                info_dict[key]['Text'] = text
                info_dict[key]['Speaker'] = 'AIHub_{}'.format(pattern_info['화자정보']['SpeakerName'])
                info_dict[key]['Gender'] = pattern_info['화자정보']['Gender']
                info_dict[key]['Emotion'] = emotion_label_dict[pattern_info['화자정보']['Emotion']]
            else:
                raise ValueError(f'Unsupported file type: {path}')

    for key in skip_info_keys:
        info_dict.pop(key, None)
    
    paths = [info['Path'] for info in info_dict.values()]
    text_dict = {
        info['Path']: info['Text']  for info in info_dict.values()
        }
    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'Korean'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        info['Path']: info['Speaker']  for info in info_dict.values()
        }
    emotion_dict = {
        info['Path']: info['Emotion']  for info in info_dict.values()
        }
    language_dict = {
        path: 'Korean'
        for path in paths
        }
    gender_dict = {
        info['Path']: info['Gender']  for info in info_dict.values()
        }

    print('AIHub info generated: {}'.format(len(paths)))
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def Basic_Info_Load(
    path: str,
    dataset_label: str,
    language: Optional[Union[str, Dict[str, str]]]= None,
    gender: Optional[Union[str, Dict[str, str]]]= None
    ):
    '''
    This is to use a customized dataset.
    path: Dataset path. In the path, there must be a 'script.txt' file. 'script.txt' must have the following structure: 'Path\tScript\tSpeaker\tEmotion'.
    dataset_label: The lable of dataset
    language: The language of dataset or speaker. When dataset is multi language, this parameter is dictionary that key and value are speaker and language, resplectly.
    gender: The gender of dataset or speaker. When dataset is multi language, this parameter is dictionary that key and value are speaker and gender, resplectly.
    '''
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            paths.append(file)
    
    text_dict = {}
    speaker_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines()[1:]:
        file, text, speaker, emotion = line.strip().split('\t')
        if (type(language) == str and language == 'English') or (type(language) == dict and language[speaker] == 'English'):
            text = unidecode(text)  # When English script, unidecode called.
        text = Text_Filtering(text)
        if text is None:
            continue
        
        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        speaker_dict[os.path.join(path, file).replace('\\', '/')] = speaker.strip()
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion.strip()

    paths = list(text_dict.keys())

    language_dict = {path: language for path in paths}

    if type(language) == str or language is None:
        language_dict = {path: language for path in paths}
    elif type(language) == dict:
        language_dict = {
            path: language[speaker]
            for path, speaker in speaker_dict.items()
            }

    if type(gender) == str or gender is None:
        gender_dict = {path: gender for path in paths}
    elif type(gender) == dict:
        gender_dict = {
            path: gender[speaker]
            for path, speaker in speaker_dict.items()
            }

    pronunciation_dict = {}
    for language in set(language_dict.values()):
        language_paths = [path for path in paths if language_dict[path] == language]
        language_pronunciations = Phonemize(
            texts= [text_dict[path] for path in language_paths],
            language= language
            )
        pronunciation_dict.update({
            path: pronunciation
            for path, pronunciation in zip(language_paths, language_pronunciations)
            })

    print('{} info generated: {}'.format(dataset_label, len(paths)))
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict


def VCTK_Info_Load(path: str):
    '''
    VCTK v0.92 is distributed as flac files.
    '''
    path = os.path.join(path, 'wav48_silence_trimmed').replace('\\', '/')
    
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            elif '_mic1' in file:
                continue

            paths.append(file)

    text_dict = {}
    for path in paths:
        if 'p315'.upper() in path.upper():  #Officially, 'p315' text is lost in VCTK dataset.
            continue
        text = Text_Filtering(unidecode(open(path.replace('wav48_silence_trimmed', 'txt').replace('flac', 'txt').replace('_mic2', ''), 'r').readlines()[0]))
        if text is None:
            continue
        
        text_dict[path] = text
            
    paths = list(text_dict.keys())
    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'English'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        path: 'VCTK.{}'.format(path.split('/')[-2].strip().upper())
        for path in paths
        }

    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}

    gender_dict = {
        'VCTK.P225': 'Female',
        'VCTK.P226': 'Male',
        'VCTK.P227': 'Male',
        'VCTK.P228': 'Female',
        'VCTK.P229': 'Female',
        'VCTK.P230': 'Female',
        'VCTK.P231': 'Female',
        'VCTK.P232': 'Male',
        'VCTK.P233': 'Female',
        'VCTK.P234': 'Female',
        'VCTK.P236': 'Female',
        'VCTK.P237': 'Male',
        'VCTK.P238': 'Female',
        'VCTK.P239': 'Female',
        'VCTK.P240': 'Female',
        'VCTK.P241': 'Male',
        'VCTK.P243': 'Male',
        'VCTK.P244': 'Female',
        'VCTK.P245': 'Male',
        'VCTK.P246': 'Male',
        'VCTK.P247': 'Male',
        'VCTK.P248': 'Female',
        'VCTK.P249': 'Female',
        'VCTK.P250': 'Female',
        'VCTK.P251': 'Male',
        'VCTK.P252': 'Male',
        'VCTK.P253': 'Female',
        'VCTK.P254': 'Male',
        'VCTK.P255': 'Male',
        'VCTK.P256': 'Male',
        'VCTK.P257': 'Female',
        'VCTK.P258': 'Male',
        'VCTK.P259': 'Male',
        'VCTK.P260': 'Male',
        'VCTK.P261': 'Female',
        'VCTK.P262': 'Female',
        'VCTK.P263': 'Male',
        'VCTK.P264': 'Female',
        'VCTK.P265': 'Female',
        'VCTK.P266': 'Female',
        'VCTK.P267': 'Female',
        'VCTK.P268': 'Female',
        'VCTK.P269': 'Female',
        'VCTK.P270': 'Male',
        'VCTK.P271': 'Male',
        'VCTK.P272': 'Male',
        'VCTK.P273': 'Male',
        'VCTK.P274': 'Male',
        'VCTK.P275': 'Male',
        'VCTK.P276': 'Female',
        'VCTK.P277': 'Female',
        'VCTK.P278': 'Male',
        'VCTK.P279': 'Male',
        'VCTK.P280': 'Female',
        'VCTK.P281': 'Male',
        'VCTK.P282': 'Female',
        'VCTK.P283': 'Male',
        'VCTK.P284': 'Male',
        'VCTK.P285': 'Male',
        'VCTK.P286': 'Male',
        'VCTK.P287': 'Male',
        'VCTK.P288': 'Female',
        'VCTK.P292': 'Male',
        'VCTK.P293': 'Female',
        'VCTK.P294': 'Female',
        'VCTK.P295': 'Female',
        'VCTK.P297': 'Female',
        'VCTK.P298': 'Male',
        'VCTK.P299': 'Female',
        'VCTK.P300': 'Female',
        'VCTK.P301': 'Female',
        'VCTK.P302': 'Male',
        'VCTK.P303': 'Female',
        'VCTK.P304': 'Male',
        'VCTK.P305': 'Female',
        'VCTK.P306': 'Female',
        'VCTK.P307': 'Female',
        'VCTK.P308': 'Female',
        'VCTK.P310': 'Female',
        'VCTK.P311': 'Male',
        'VCTK.P312': 'Female',
        'VCTK.P313': 'Female',
        'VCTK.P314': 'Female',
        'VCTK.P316': 'Male',
        'VCTK.P317': 'Female',
        'VCTK.P318': 'Female',
        'VCTK.P323': 'Female',
        'VCTK.P326': 'Male',
        'VCTK.P329': 'Female',
        'VCTK.P330': 'Female',
        'VCTK.P333': 'Female',
        'VCTK.P334': 'Male',
        'VCTK.P335': 'Female',
        'VCTK.P336': 'Female',
        'VCTK.P339': 'Female',
        'VCTK.P340': 'Female',
        'VCTK.P341': 'Female',
        'VCTK.P343': 'Female',
        'VCTK.P345': 'Male',
        'VCTK.P347': 'Male',
        'VCTK.P351': 'Female',
        'VCTK.P360': 'Male',
        'VCTK.P361': 'Female',
        'VCTK.P362': 'Female',
        'VCTK.P363': 'Male',
        'VCTK.P364': 'Male',
        'VCTK.P374': 'Male',
        'VCTK.P376': 'Male',
        'VCTK.S5': 'Female',
        }
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print('VCTK info generated: {}'.format(len(paths)))

    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def Libri_Info_Load(path: str):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file_path)[1].upper() in using_extension:
                continue
            paths.append(file_path)

    text_dict = {}
    for file_path in paths:
        text = Text_Filtering(unidecode(open('{}.normalized.txt'.format(os.path.splitext(file_path)[0]), 'r', encoding= 'utf-8-sig').readlines()[0]))
        if text is None:
            continue
        
        text_dict[file_path] = text

    paths = list(text_dict.keys())
    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'English'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        path: 'Libri.{:04d}'.format(int(path.split('/')[-3].strip().upper()))
        for path in paths
        }

    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}

    gender_dict = {}
    for line in open(os.path.join(os.path.join(path, 'SPEAKERS.txt').replace('\\', '/')), 'r', encoding= 'utf-8-sig').readlines()[12:]:
        speaker, gender, *_ = [x.strip() for x in line.strip().split('|')]
        gender_dict[f'Libri.{int(speaker):04d}'] = 'Male' if gender == 'M' else 'Female'
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print('Libri info generated: {}'.format(len(paths)))
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def LJ_Info_Load(path: str):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            paths.append(file)

    text_dict = {}
    for line in open(os.path.join(path, 'metadata.csv').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines():
        line = line.strip().split('|')        
        text = Text_Filtering(unidecode(line[2].strip()))
        if text is None:
            continue
        wav_path = os.path.join(path, 'wavs', '{}.wav'.format(line[0]))
        
        text_dict[wav_path] = text

    paths = list(text_dict.keys())

    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'English'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}
    
    speaker_dict = {
        path: 'LJ'
        for path in paths
        }
    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}
    gender_dict = {path: 'Female' for path in paths}

    print('LJ info generated: {}'.format(len(paths)))
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def MLS_Info_Load(path: str):
    paths, texts, speakers = [], [], []
    for partition in ['dev', 'test', 'train']:
        for line in open(os.path.join(path, partition, 'transcripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines():
            file_path, text = line.strip().split('\t')
            speaker, book_id, _ = file_path.strip().split('_')
            file_path = os.path.join(path, partition, 'audio', speaker, book_id, f'{file_path}.opus')
            if not os.path.exists(file_path):
                continue                        
            text = Text_Filtering(unidecode(text))
            if text is None:
                continue            
            paths.append(file_path)
            texts.append(text)
            speakers.append(int(speaker))

    text_dict = {
        path: text
        for path, text in zip(paths, texts)
        }
    
    pronunciations = Phonemize(
        texts= texts,
        language= 'English'
        )    
    pronunciation_dict = {
        path: pronunciation
        for path, pronunciation in zip(paths, pronunciations)
        }

    speaker_dict = {
        path: f'MLS.{speaker:05d}'
        for path, speaker in zip(paths, speakers)
        }
    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}
    
    gender_dict = {}
    for line in open(os.path.join(os.path.join(path, 'metainfo.txt').replace('\\', '/')), 'r', encoding= 'utf-8-sig').readlines()[1:]:
        speaker, gender, *_ = [x.strip() for x in line.strip().split('|')]
        gender_dict[int(speaker)] = 'Male' if gender == 'M' else 'Female'
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in zip(paths, speakers)
        }

    print('MLS info generated: {}'.format(len(paths)))
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def Split_Eval(paths: List[str], eval_ratio: float= 0.001, min_eval: int= 1):
    shuffle(paths)
    index = max(int(len(paths) * eval_ratio), min_eval)
    return paths[index:], paths[:index]

def Metadata_Generate(eval: bool= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path
    metadata_File = hp.Train.Eval_Pattern.Metadata_File if eval else hp.Train.Train_Pattern.Metadata_File

    latent_dict = {}
    mel_dict = {}
    f0_dict = {}
    speakers = []
    emotions = []
    languages = []
    genders = []
    language_and_gender_dict_by_speaker = {}

    new_metadata_dict = {
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Latent_Length_Dict': {},
        'Mel_Length_Dict': {},
        'F0_Length_Dict': {},
        'Speaker_Dict': {},
        'Emotion_Dict': {},
        'Dataset_Dict': {},
        'File_List_by_Speaker_Dict': {},
        'Text_Length_Dict': {},
        }

    files_tqdm = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path, followlinks=True)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path, followlinks=True):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_dict = pickle.load(f)

            file = os.path.join(root, file).replace("\\", "/").replace(pattern_path, '').lstrip('/')

            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Audio', 'Latent', 'F0', 'Speaker', 'Emotion', 'Language', 'Gender', 'Dataset', 'Text', 'Pronunciation')
                    ]):
                    continue
                new_metadata_dict['Audio_Length_Dict'][file] = pattern_dict['Audio'].shape[0]
                new_metadata_dict['Latent_Length_Dict'][file] = pattern_dict['Latent'].shape[1]
                new_metadata_dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[1]
                new_metadata_dict['F0_Length_Dict'][file] = pattern_dict['F0'].shape[0]
                new_metadata_dict['Speaker_Dict'][file] = pattern_dict['Speaker']
                new_metadata_dict['Emotion_Dict'][file] = pattern_dict['Emotion']
                new_metadata_dict['Dataset_Dict'][file] = pattern_dict['Dataset']
                new_metadata_dict['File_List'].append(file)
                if not pattern_dict['Speaker'] in new_metadata_dict['File_List_by_Speaker_Dict'].keys():
                    new_metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']] = []
                new_metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']].append(file)
                new_metadata_dict['Text_Length_Dict'][file] = len(pattern_dict['Text'])

                if not pattern_dict['Speaker'] in latent_dict.keys():
                    latent_dict[pattern_dict['Speaker']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Speaker'] in mel_dict.keys():
                    mel_dict[pattern_dict['Speaker']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Speaker'] in f0_dict.keys():
                    f0_dict[pattern_dict['Speaker']] = []
                
                latent = hificodec.quantizer.embed(torch.from_numpy(pattern_dict['Latent']).T[None].long().to(device)).squeeze(0).cpu()
                latent_dict[pattern_dict['Speaker']]['Min'] = min(latent_dict[pattern_dict['Speaker']]['Min'], latent.min().item())
                latent_dict[pattern_dict['Speaker']]['Max'] = max(latent_dict[pattern_dict['Speaker']]['Max'], latent.max().item())
                mel_dict[pattern_dict['Speaker']]['Min'] = min(mel_dict[pattern_dict['Speaker']]['Min'], pattern_dict['Mel'].min().item())
                mel_dict[pattern_dict['Speaker']]['Max'] = max(mel_dict[pattern_dict['Speaker']]['Max'], pattern_dict['Mel'].max().item())

                f0_dict[pattern_dict['Speaker']].append(pattern_dict['F0'])
                speakers.append(pattern_dict['Speaker'])
                emotions.append(pattern_dict['Emotion'])
                languages.append(pattern_dict['Language'])
                genders.append(pattern_dict['Gender'])
                language_and_gender_dict_by_speaker[pattern_dict['Speaker']] = {
                    'Language': pattern_dict['Language'],
                    'Gender': pattern_dict['Gender']
                    }
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

            files_tqdm.update(1)

    with open(os.path.join(pattern_path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    if not eval:
        yaml.dump(
            latent_dict,
            open(hp.Latent_Info_Path, 'w')
            )
        yaml.dump(
            mel_dict,
            open(hp.Mel_Info_Path, 'w')
            )

        f0_info_dict = {}
        for speaker, f0_list in f0_dict.items():
            f0 = np.hstack(f0_list)
            f0 = np.clip(f0, 0, np.inf)
            f0 = f0[f0 != 0.0].astype(np.float32)
            f0_info_dict[speaker] = {
                'Min': f0.min().item(),
                'Max': f0.max().item(),
                'Mean': f0.mean().item(),
                'Std': f0.std().item()
                }
        yaml.dump(
            f0_info_dict,
            open(hp.F0_Info_Path, 'w')
            )
        
        speaker_index_dict = {
            speaker: index
            for index, speaker in enumerate(sorted(set(speakers)))
            }
        yaml.dump(
            speaker_index_dict,
            open(hp.Speaker_Info_Path, 'w')
            )
            
        emotion_index_dict = {
            emotion: index
            for index, emotion in enumerate(sorted(set(emotions)))
            }
        yaml.dump(
            emotion_index_dict,
            open(hp.Emotion_Info_Path, 'w')
            )

        language_index_dict = {
            language: index
            for index, language in enumerate(sorted(set(languages)))
            }
        yaml.dump(
            language_index_dict,
            open(hp.Language_Info_Path, 'w')
            )

        gender_index_dict = {
            gender: index
            for index, gender in enumerate(sorted(set(genders)))
            }
        yaml.dump(
            gender_index_dict,
            open(hp.Gender_Info_Path, 'w')
            )

        yaml.dump(
            language_and_gender_dict_by_speaker,
            open(hp.Language_and_Gender_Info_by_Speaker_Path, 'w')
            )

    print('Metadata generate done.')

def Token_dict_Generate(tokens: Union[List[str], str]):
    tokens = ['<S>', '<E>', '<P>'] + sorted(list(tokens))        
    token_dict = {token: index for index, token in enumerate(tokens)}
    
    os.makedirs(os.path.dirname(hp.Token_Path), exist_ok= True)    
    yaml.dump(token_dict, open(hp.Token_Path, 'w', encoding='utf-8-sig'), allow_unicode= True)

    return token_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hp", "--hyper_parameters", required=True, type= str)
    parser.add_argument("-selvas", "--selvas_path", required=False)
    parser.add_argument("-kss", "--kss_path", required=False)
    parser.add_argument("-aihub", "--aihub_path", required=False)

    parser.add_argument("-vctk", "--vctk_path", required=False)
    parser.add_argument("-libri", "--libri_path", required=False)
    parser.add_argument("-lj", "--lj_path", required=False)
    parser.add_argument("-mls", "--mls_path", required=False)

    parser.add_argument("-evalr", "--eval_ratio", default= 0.001, type= float)
    parser.add_argument("-evalm", "--eval_min", default= 1, type= int)
    parser.add_argument("-mw", "--max_worker", default= 2, required=False, type= int)

    args = parser.parse_args()

    global hp
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    train_paths, eval_paths = [], []
    text_dict = {}
    pronunciation_dict = {}
    speaker_dict = {}
    emotion_dict = {}
    language_dict = {}
    gender_dict = {}
    dataset_dict = {}
    tag_dict = {}

    if not args.selvas_path is None:
        selvas_paths, selvas_text_dict, selvas_pronunciation_dict, selvas_speaker_dict, selvas_emotion_dict, selvas_language_dict, selvas_gender_dict = Selvas_Info_Load(path= args.selvas_path)
        selvas_paths = Split_Eval(selvas_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(selvas_paths[0])
        eval_paths.extend(selvas_paths[1])
        text_dict.update(selvas_text_dict)
        pronunciation_dict.update(selvas_pronunciation_dict)
        speaker_dict.update(selvas_speaker_dict)
        emotion_dict.update(selvas_emotion_dict)
        language_dict.update(selvas_language_dict)
        gender_dict.update(selvas_gender_dict)
        dataset_dict.update({path: 'Selvas' for paths in selvas_paths for path in paths})
        tag_dict.update({path: '' for paths in selvas_paths for path in paths})

    if not args.kss_path is None:
        kss_paths, kss_text_dict, kss_pronunciation_dict, kss_speaker_dict, kss_emotion_dict, kss_language_dict, kss_gender_dict = KSS_Info_Load(path= args.kss_path)
        kss_paths = Split_Eval(kss_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(kss_paths[0])
        eval_paths.extend(kss_paths[1])
        text_dict.update(kss_text_dict)
        pronunciation_dict.update(kss_pronunciation_dict)
        speaker_dict.update(kss_speaker_dict)
        emotion_dict.update(kss_emotion_dict)
        language_dict.update(kss_language_dict)
        gender_dict.update(kss_gender_dict)
        dataset_dict.update({path: 'KSS' for paths in kss_paths for path in paths})
        tag_dict.update({path: '' for paths in kss_paths for path in paths})

    if not args.aihub_path is None:
        aihub_paths, aihub_text_dict, aihub_pronunciation_dict, aihub_speaker_dict, aihub_emotion_dict, aihub_language_dict, aihub_gender_dict = AIHub_Info_Load(
            path= args.aihub_path
            )
        aihub_paths = Split_Eval(aihub_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(aihub_paths[0])
        eval_paths.extend(aihub_paths[1])
        text_dict.update(aihub_text_dict)
        pronunciation_dict.update(aihub_pronunciation_dict)
        speaker_dict.update(aihub_speaker_dict)
        emotion_dict.update(aihub_emotion_dict)
        language_dict.update(aihub_language_dict)
        gender_dict.update(aihub_gender_dict)
        dataset_dict.update({path: 'AIHub' for paths in aihub_paths for path in paths})
        tag_dict.update({path: '' for paths in aihub_paths for path in paths})



    if not args.vctk_path is None:
        vctk_paths, vctk_text_dict, vctk_pronunciation_dict, vctk_speaker_dict, vctk_emotion_dict, vctk_language_dict, vctk_gender_dict = VCTK_Info_Load(path= args.vctk_path)
        vctk_paths = Split_Eval(vctk_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(vctk_paths[0])
        eval_paths.extend(vctk_paths[1])
        text_dict.update(vctk_text_dict)
        pronunciation_dict.update(vctk_pronunciation_dict)
        speaker_dict.update(vctk_speaker_dict)
        emotion_dict.update(vctk_emotion_dict)
        language_dict.update(vctk_language_dict)
        gender_dict.update(vctk_gender_dict)
        dataset_dict.update({path: 'VCTK' for paths in vctk_paths for path in paths})
        tag_dict.update({path: '' for paths in vctk_paths for path in paths})

    if not args.libri_path is None:
        libri_paths, libri_text_dict, libri_pronunciation_dict, libri_speaker_dict, libri_emotion_dict, libri_language_dict, libri_gender_dict = Libri_Info_Load(path= args.libri_path)
        libri_paths = Split_Eval(libri_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(libri_paths[0])
        eval_paths.extend(libri_paths[1])
        text_dict.update(libri_text_dict)
        pronunciation_dict.update(libri_pronunciation_dict)
        speaker_dict.update(libri_speaker_dict)
        emotion_dict.update(libri_emotion_dict)
        language_dict.update(libri_language_dict)
        gender_dict.update(libri_gender_dict)
        dataset_dict.update({path: 'Libri' for paths in libri_paths for path in paths})
        tag_dict.update({path: '' for paths in libri_paths for path in paths})

    if not args.lj_path is None:
        lj_paths, lj_text_dict, lj_pronunciation_dict, lj_speaker_dict, lj_emotion_dict, lj_language_dict, lj_gender_dict = LJ_Info_Load(path= args.lj_path)
        lj_paths = Split_Eval(lj_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(lj_paths[0])
        eval_paths.extend(lj_paths[1])
        text_dict.update(lj_text_dict)
        pronunciation_dict.update(lj_pronunciation_dict)
        speaker_dict.update(lj_speaker_dict)
        emotion_dict.update(lj_emotion_dict)
        language_dict.update(lj_language_dict)
        gender_dict.update(lj_gender_dict)
        dataset_dict.update({path: 'LJ' for paths in lj_paths for path in paths})
        tag_dict.update({path: '' for paths in lj_paths for path in paths})

    if not args.mls_path is None:
        mls_paths, mls_text_dict, mls_pronunciation_dict, mls_speaker_dict, mls_emotion_dict, mls_language_dict, mls_gender_dict = MLS_Info_Load(path= args.mls_path)
        mls_paths = Split_Eval(mls_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(mls_paths[0])
        eval_paths.extend(mls_paths[1])
        text_dict.update(mls_text_dict)
        pronunciation_dict.update(mls_pronunciation_dict)
        speaker_dict.update(mls_speaker_dict)
        emotion_dict.update(mls_emotion_dict)
        language_dict.update(mls_language_dict)
        gender_dict.update(mls_gender_dict)
        dataset_dict.update({path: 'MLS' for paths in mls_paths for path in paths})
        tag_dict.update({path: '' for paths in mls_paths for path in paths})

    # if len(train_paths) == 0 or len(eval_paths) == 0:
    #     raise ValueError('Total info count must be bigger than 0.')

    train_paths = sorted(train_paths)
    eval_paths = sorted(eval_paths)

    tokens = set([
        token
        for phonemes in pronunciation_dict.values()
        for token in phonemes
        ])
    token_dict = Token_dict_Generate(tokens= tokens)

    with PE(max_workers = args.max_worker) as pe:
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [
                    (
                        path,
                        speaker_dict[path],
                        emotion_dict[path],
                        language_dict[path],
                        gender_dict[path],
                        dataset_dict[path],
                        text_dict[path],
                        pronunciation_dict[path],
                        tag_dict[path],
                        False
                        )
                    for path in train_paths
                    ]
                ),
            total= len(train_paths)
            ):
            pass
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [
                    (
                        path,
                        speaker_dict[path],
                        emotion_dict[path],
                        language_dict[path],
                        gender_dict[path],
                        dataset_dict[path],
                        text_dict[path],
                        pronunciation_dict[path],
                        tag_dict[path],
                        True
                        )
                    for path in eval_paths
                    ]
                ),
            total= len(eval_paths)
            ):
            pass

    Metadata_Generate()
    Metadata_Generate(eval= True)

# python Pattern_Generator.py -hp Hyper_Parameters.yaml -lj D:\Rawdata\LJSpeech
# python Pattern_Generator.py -hp Hyper_Parameters.yaml -vctk D:\Rawdata\VCTK092
# python Pattern_Generator.py -hp Hyper_Parameters.yaml -mls D:\Rawdata\mls_english_opus
# python Pattern_Generator.py -hp Hyper_Parameters.yaml -libri D:\Rawdata\LibriTTS