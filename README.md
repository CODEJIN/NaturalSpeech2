# NaturalSpeech 2 (WIP)

* This code is a unofficial implementation of NaturalSpeech 2.
* The algorithm is based on the following paper:

```
Shen, K., Ju, Z., Tan, X., Liu, Y., Leng, Y., He, L., ... & Bian, J. (2023). NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers. arXiv preprint arXiv:2304.09116.
```

# Modifications from Paper
* The structure is derived from NaturalSpeech 2, but I made several modifications.
* About CE-RVQ
    * The CE-RVQ implementation in the current repository is incomplete.
        * I had doubts about the loss calculation formula mentioned in the paper, so the previous implementation has been commented out.
        * I think the current implementation aligns with the purpose of CE-RVQ, but deviates from the paper. 
        * The current implementation has not been verified how positively this loss contributes to model training.
        * I would greatly appreciate any advice or suggestions you may have regarding this matter.
    * CE-RVQ is only applied to the last RVQ layer.
        * In the tests, when applied to all layers, the loss term of CE-RVQ becomes too large, causing the data loss to not decrease properly.
        * The cause of this issue is unknown.
        * One possible explanation is that the HifiCodec used in this repository, which has only two RVQ layers, may have a significant influence from the former RVQ layers compared to the 16 RVQ in the original paper.
* The audio codec has been changed to `HifiCodec` from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
    * This is done to reduce the time spent training a separate audio codec.
    * The model uses 22.05Khz audio, but no audio resampling is applied.
    * To maintain similarity with the paper, it may be better to apply Google's `SoundStream` instead of HifiCodec, but I couldn't apply SoundStream to this repository because official pyTorch source code or pretrained model was not provided.
        * Although this repository does not use, there is also a [c++ or tflite version of Lyra](https://github.com/google/lyra), which may allow the application of SoundStream using it.
    * Meta's Encodec 24K version was also tested, but it could not be trained.
* Information on the segment length σ of the speech prompt during training was not found in the paper and was arbitrarily set.
    * The `σ` = 3, 5, and 10 seconds used in the evaluation of paper are too long to apply to both the variance predictor and diffusion during training.
    * To ensure stability in pattern usage, half the length of the shortest pattern used in each training is set as `σ` for each training.
* The target duration is obtained through `Alignment learning framework (ALF)`, rather than being brought in externally.
    * Using external modules such as Montreal Force Alignment (MFA) may have benefits in terms of training speed or stability, but I prioritized simplifying the training process.    
* Padding is applied between tokens like `'A <P> B <P> C ....'`
    * I could not verify whether there was a difference in performance depending on its usage.    


# Supported dataset
* To apply zero-shot reported in the paper, I believe that it is necessary to have as many speakers as possible in the training data, but I were unable to test [Multilingual LibriSpeech](https://www.openslr.org/94/) due to current environment.
* Tested
    * [LJ Dataset](https://keithito.com/LJ-Speech-Dataset/)
    * [VCTK Dataset](https://datashare.ed.ac.uk/handle/10283/2651)
        * This repository used the VCTK092 from Torchaudio(https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip)
* Supported but not tested
    * [LibriTTS Dataset](https://www.openslr.org/60/)
    * [Mulitilingual LibriSpeech Dataset](https://www.openslr.org/94/)
        * Only English language dataset generation is tested.

# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in [Hyper_Parameters.yaml](Hyper_Parameters.yaml) according to your environment.

* Sound
    * Setting basic sound parameters.

* Tokens
    * The number of token.    
    * After pattern generating, you can see which tokens are included in the dataset at `Token_Path`.

* Audio_Codec
    * Setting the audio codec.
    * This repository is using HifiCodec, so only the size of the latents output from HifiCodec's encoder is set for reference in other modules.

* Train
    * Setting the parameters of training.

* Inference_Batch_Size
    * Setting the batch size when inference

* Inference_Path
    * Setting the inference path

* Checkpoint_Path
    * Setting the checkpoint path

* Log_Path
    * Setting the tensorboard log path

* Use_Mixed_Precision
    * Setting using mixed precision

* Use_Multi_GPU
    * Setting using multi gpu
    * By the nvcc problem, Only linux supports this option.
    * If this is `True`, device parameter is also multiple like `0,1,2,3`.
    * And you have to change the training command also: please check [multi_gpu.sh](./multi_gpu.sh).

* Device
    * Setting which GPU devices are used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'. (But, I don't recommend while training.)

# Generate pattern

```
python Pattern_Generate.py [parameters]
```
## Parameters
* -lj
    * The path of LJSpeech dataset
* -vctk
    * The path of VCTK dataset
* -libri
    * The path of LbiriTTS dataset
* -hp
    * The path of hyperparameter.

## About phonemizer
* To phoneme string generate, this repository uses phonimizer library.
* Please refer [here](https://bootphon.github.io/phonemizer/install.html) to install phonemizer and backend
* In Windows, you need more setting to use phonemizer.
    * Please refer [here](https://github.com/bootphon/phonemizer/issues/44)
    * In conda enviornment, the following commands are useful.
        ```bash
        conda env config vars set PHONEMIZER_ESPEAK_PATH='C:\Program Files\eSpeak NG'
        conda env config vars set PHONEMIZER_ESPEAK_LIBRARY='C:\Program Files\eSpeak NG\libespeak-ng.dll'
        ```
# Run

## Command

### Single GPU
```
python Train.py -hp <path> -s <int>
```

* `-hp <path>`
    * The hyper paramter file path
    * This is required.

* `-s <int>`
    * The resume step parameter.
    * Default is `0`.
    * If value is `0`, model try to search the latest checkpoint.

### Multi GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=32 python -m torch.distributed.launch --nproc_per_node=8 Train.py --hyper_parameters Hyper_Parameters.yaml --port 54322
```

* I recommend to check the [multi_gpu.sh](./multi_gpu.sh).

# TODO
* Verification
