import numpy as np
from scipy import signal
import librosa


def Audio_Prep(path, sample_rate, trim_top_db= 60):
    audio = librosa.load(path, sr= sample_rate)[0]
    audio = librosa.effects.trim(audio, top_db=trim_top_db, frame_length= 512, hop_length= 256)[0]
    audio = librosa.util.normalize(audio)

    return audio


def Mel_Generate(
    audio,
    sample_rate,
    num_mel,
    num_frequency,
    window_length,
    hop_length,
    pre_emphasis= 0.97,    
    mel_fmin= 125,
    mel_fmax= 7600,
    min_level_db= -100,   
    max_abs_value= 4.0
    ):
    pre_emphasis_audio = Preemphasis(audio, pre_emphasis= pre_emphasis)
    
    n_fft = (num_frequency - 1) * 2
    magnitude = np.abs(librosa.stft(
        y= pre_emphasis_audio,
        n_fft= n_fft,
        hop_length= hop_length,
        win_length= window_length
        ))

    mel_filter = librosa.filters.mel(sr= sample_rate, n_fft= n_fft, n_mels= num_mel, fmin= mel_fmin, fmax= mel_fmax)
    magnitude = mel_filter @ magnitude
    
    db = 20 * np.log10(magnitude + 1e-7)
    mel = np.clip(
        (2 * max_abs_value) * (db - min_level_db)/-min_level_db - max_abs_value,
        -max_abs_value,
        max_abs_value
        ).T
    
    return mel 


def Preemphasis(audio, pre_emphasis = 0.97):
    return signal.lfilter([1.0, -pre_emphasis], [1.0], audio)

