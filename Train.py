from email.policy import strict
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, pickle, wandb
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Modules.Modules import NaturalSpeech2, Mask_Generate
from Modules.Nvidia_Alignment_Learning_Framework import AttentionBinarizationLoss, AttentionCTCLoss

from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Noam_Scheduler import Noam_Scheduler
from Logger import Logger

from meldataset import mel_spectrogram
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict


import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.gpu_id == 0:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }

            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp)
                    )
                wandb.watch(self.model)

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
        f0_info_dict = yaml.load(open(self.hp.F0_Info_Path, 'r'), Loader=yaml.Loader)

        train_dataset = Dataset(
            token_dict= token_dict,
            f0_info_dict= f0_info_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            latent_length_min= max(self.hp.Train.Train_Pattern.Feature_Length.Min, self.hp.Train.Segment_Size),
            latent_length_max= self.hp.Train.Train_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Train_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Train_Pattern.Text_Length.Max,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            f0_info_dict= f0_info_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            latent_length_min= max(self.hp.Train.Train_Pattern.Feature_Length.Min, self.hp.Train.Segment_Size),
            latent_length_max= self.hp.Train.Eval_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Eval_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Eval_Pattern.Text_Length.Max,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            sample_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Frame_Shift,
            texts= self.hp.Train.Inference_in_Train.Text,
            references= self.hp.Train.Inference_in_Train.Reference,
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(
            token_dict= token_dict
            )
        inference_collater = Inference_Collater(
            token_dict= token_dict,
            speech_prompt_length= self.hp.Train.Inference_in_Train.Speech_Prompt_Length
            )

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.DistributedSampler(eval_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model = NaturalSpeech2(self.hp).to(self.device)
        self.criterion_dict = {
            'MSE': torch.nn.MSELoss(reduction= 'none').to(self.device),
            'MAE': torch.nn.L1Loss(reduction= 'none').to(self.device),
            'CE': torch.nn.CrossEntropyLoss(reduction= 'none').to(self.device),
            'Attention_Binarization': AttentionBinarizationLoss(),
            'Attention_CTC': AttentionCTCLoss(),
            }
        self.optimizer = torch.optim.AdamW(
            params= self.model.parameters(),
            lr= self.hp.Train.Learning_Rate.Initial,
            betas= (self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
            eps= self.hp.Train.ADAM.Epsilon
            )
        self.scheduler = Noam_Scheduler(
            optimizer= self.optimizer,
            warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step
            )

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        # if self.gpu_id == 0:
        #     logging.info(self.model)

    def Train_Step(self, tokens, token_lengths, speech_prompts, speech_prompts_for_diffusion, latents, latent_lengths, f0s, mels, attention_priors):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        speech_prompts = speech_prompts.to(self.device, non_blocking=True)
        speech_prompts_for_diffusion = speech_prompts_for_diffusion.to(self.device, non_blocking=True)
        latents = latents.to(self.device, non_blocking=True)
        latent_lengths = latent_lengths.to(self.device, non_blocking=True)
        f0s = f0s.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        attention_priors = attention_priors.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            _, diffusion_targets, diffusion_predictions, \
            duration_predictions, f0_predictions, ce_rvq_targets, ce_rvq_logits, \
            attention_softs, attention_hards, attention_logprobs, durations, _, _ = self.model(
                tokens= tokens,
                token_lengths= token_lengths,
                speech_prompts= speech_prompts,
                speech_prompts_for_diffusion= speech_prompts_for_diffusion,
                latents= latents,
                latent_lengths= latent_lengths,
                f0s= f0s,
                mels= mels,
                attention_priors= attention_priors
                )
            
            with torch.cuda.amp.autocast(enabled= False):
                token_masks = (~Mask_Generate(
                    lengths= token_lengths,
                    max_length= tokens.size(1)
                    ).to(tokens.device)).float()
                latent_masks = (~Mask_Generate(
                    lengths= latent_lengths,
                    max_length= latents.size(2)
                    ).to(latents.device)).float()
                
                loss_dict['Diffusion'] = self.criterion_dict['MSE'](
                    diffusion_targets,
                    diffusion_predictions,
                    ).mean()
                loss_dict['Duration'] = (self.criterion_dict['MAE'](
                    duration_predictions,
                    durations
                    ) * token_masks).sum() / token_masks.sum()
                loss_dict['F0'] = (self.criterion_dict['MAE'](
                    f0_predictions.float(),
                    f0s
                    ) * latent_masks).sum() / latent_masks.sum()
                loss_dict['CE_RVQ'] = self.criterion_dict['CE'](
                    ce_rvq_logits,
                    ce_rvq_targets
                    ).mean()
                
                loss_dict['Attention_Binarization'] = self.criterion_dict['Attention_Binarization'](attention_hards, attention_softs)
                loss_dict['Attention_CTC'] = self.criterion_dict['Attention_CTC'](attention_logprobs, token_lengths, latent_lengths)

        self.optimizer.zero_grad()
        self.scaler.scale(
            loss_dict['Diffusion'] +
            loss_dict['Duration'] +
            loss_dict['F0'] +
            loss_dict['Attention_Binarization'] +
            loss_dict['Attention_CTC']
            ).backward()

        self.scaler.unscale_(self.optimizer)

        if self.hp.Train.Gradient_Norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.steps += 1
        self.tqdm.update(1)
        
        self.scheduler.step()

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for tokens, token_lengths, speech_prompts, speech_prompts_for_diffusion, latents, latent_lengths, f0s, mels, attention_priors in self.dataloader_dict['Train']:
            self.Train_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                speech_prompts= speech_prompts,
                speech_prompts_for_diffusion= speech_prompts_for_diffusion,
                latents= latents,
                latent_lengths= latent_lengths,
                f0s= f0s,
                mels= mels,
                attention_priors= attention_priors,
                )

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    def Evaluation_Step(self, tokens, token_lengths, speech_prompts, speech_prompts_for_diffusion, latents, latent_lengths, f0s, mels, attention_priors):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        speech_prompts = speech_prompts.to(self.device, non_blocking=True)
        speech_prompts_for_diffusion = speech_prompts_for_diffusion.to(self.device, non_blocking=True)
        latents = latents.to(self.device, non_blocking=True)
        latent_lengths = latent_lengths.to(self.device, non_blocking=True)
        f0s = f0s.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        attention_priors = attention_priors.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            _, diffusion_targets, diffusion_predictions, \
            duration_predictions, f0_predictions, ce_rvq_targets, ce_rvq_logits, \
            attention_softs, attention_hards, attention_logprobs, durations, _, _ = self.model(
                tokens= tokens,
                token_lengths= token_lengths,
                speech_prompts= speech_prompts,
                speech_prompts_for_diffusion= speech_prompts_for_diffusion,
                latents= latents,
                latent_lengths= latent_lengths,
                f0s= f0s,
                mels= mels,
                attention_priors= attention_priors
                )

            with torch.cuda.amp.autocast(enabled= False):
                token_masks = (~Mask_Generate(
                    lengths= token_lengths,
                    max_length= tokens.size(1)
                    ).to(tokens.device)).float()
                latent_masks = (~Mask_Generate(
                    lengths= latent_lengths,
                    max_length= latents.size(2)
                    ).to(latents.device)).float()
                
                loss_dict['Diffusion'] = self.criterion_dict['MSE'](
                    diffusion_targets,
                    diffusion_predictions,
                    ).mean()
                loss_dict['Duration'] = (self.criterion_dict['MAE'](
                    duration_predictions,
                    durations
                    ) * token_masks).sum() / token_masks.sum()
                loss_dict['F0'] = (self.criterion_dict['MAE'](
                    f0_predictions.float(),
                    f0s
                    ) * latent_masks).sum() / latent_masks.sum()
                loss_dict['CE_RVQ'] = self.criterion_dict['CE'](
                    ce_rvq_logits,
                    ce_rvq_targets
                    ).mean()
                loss_dict['Attention_Binarization'] = self.criterion_dict['Attention_Binarization'](attention_hards, attention_softs)
                loss_dict['Attention_CTC'] = self.criterion_dict['Attention_CTC'](attention_logprobs, token_lengths, latent_lengths)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

        return durations

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, (tokens, token_lengths, speech_prompts, speech_prompts_for_diffusion, latents, latent_lengths, f0s, mels, attention_priors) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            durations = self.Evaluation_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                speech_prompts= speech_prompts,
                speech_prompts_for_diffusion= speech_prompts_for_diffusion,
                latents= latents,
                latent_lengths= latent_lengths,
                f0s= f0s,
                mels= mels,
                attention_priors= attention_priors,
                )

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_dict['Evaluation'].add_histogram_model(self.model, 'NaturalSpeech2', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, tokens.size(0))

            with torch.inference_mode():
                prediction_audios, *_, prediction_durations, prediction_f0s, prediction_latent_lengths = self.model(
                    tokens= tokens[index].unsqueeze(0).to(self.device),
                    token_lengths= token_lengths[index].unsqueeze(0).to(self.device),
                    speech_prompts= speech_prompts[index].unsqueeze(0).to(self.device),
                    ddim_steps= max(self.hp.Diffusion.Max_Step // 10, 100)
                    )
                target_audios = self.model.encodec.decode([[latents[index].unsqueeze(0).to(self.device), None]]).squeeze(1)
            
            token_length = token_lengths[index].item()
            target_latent_length = latent_lengths[index].item()
            prediction_latent_length = prediction_latent_lengths[0].item()
            target_audio_length = target_latent_length * self.hp.Sound.Frame_Shift
            prediction_audio_length = prediction_latent_length * self.hp.Sound.Frame_Shift

            target_audio = target_audios[0, :target_audio_length].clamp(-1.0, 1.0)
            prediction_audio = prediction_audios[0, :prediction_audio_length].clamp(-1.0, 1.0)

            target_feature = mel_spectrogram(
                target_audio.unsqueeze(0),
                n_fft= self.hp.Sound.Frame_Shift * 4,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Shift * 4,
                fmin= 0,
                fmax= None
                ).squeeze(0).cpu().numpy()
            if prediction_audio_length > self.hp.Sound.Frame_Shift * 10:
                prediction_feature = mel_spectrogram(
                    prediction_audio.unsqueeze(0),
                    n_fft= self.hp.Sound.Frame_Shift * 4,
                    num_mels= self.hp.Sound.Mel_Dim,
                    sampling_rate= self.hp.Sound.Sample_Rate,
                    hop_size= self.hp.Sound.Frame_Shift,
                    win_size= self.hp.Sound.Frame_Shift * 4,
                    fmin= 0,
                    fmax= None
                    ).squeeze(0).cpu().numpy()
            else:
                logging.warning('Prediction feature could not be generated because too shoart audio length.')
                prediction_feature = np.zeros(shape= (self.hp.Sound.Mel_Dim, 1), dtype= np.float32)

            target_audio = target_audio.cpu().numpy()
            prediction_audio = prediction_audio.cpu().numpy()

            target_f0 = f0s[index, :target_latent_length].cpu().numpy() 
            prediction_f0 = prediction_f0s[0, :prediction_latent_length].cpu().numpy()

            target_duration = durations[index, :token_length].long()
            target_duration = torch.arange(target_duration.size(0)).repeat_interleave(target_duration.cpu())[:prediction_latent_length].cpu().numpy()

            prediction_duration = prediction_durations[0, :token_length]
            prediction_duration = torch.arange(prediction_duration.size(0)).repeat_interleave(prediction_duration.cpu())[:prediction_latent_length].cpu().numpy()

            image_dict = {
                'Feature/Target': (target_feature, None, 'auto', None, None, None),
                'Feature/Prediction': (prediction_feature, None, 'auto', None, None, None),
                'Duration/Target': (target_duration, None, 'auto', None, None, None),
                'Duration/Prediction': (prediction_duration, None, 'auto', None, None, None),
                'F0/Target': (target_f0, None, 'auto', None, None, None),
                'F0/Prediction': (prediction_f0, None, 'auto', None, None, None),
                }
            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Linear': (prediction_audio, self.hp.Sound.Sample_Rate),
                }

            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)
            self.writer_dict['Evaluation'].add_audio_dict(audio_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {
                        'Evaluation.Feature.Target': wandb.Image(target_feature),
                        'Evaluation.Feature.Prediction': wandb.Image(prediction_feature),
                        'Evaluation.F0': wandb.plot.line_series(
                            xs= np.arange(target_f0.shape[0]),
                            ys= [target_f0, prediction_f0],
                            keys= ['Target', 'Prediction'],
                            title= 'F0',
                            xname= 'Latent_t'
                            ),
                        'Evaluation.Duration': wandb.plot.line_series(
                            xs= np.arange(prediction_latent_length),
                            ys= [target_duration, prediction_duration],
                            keys= ['Target', 'Prediction'],
                            title= 'Duration',
                            xname= 'Latent_t'
                            ),
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Target_Audio'
                            ),
                        'Evaluation.Audio.Linear': wandb.Audio(
                            prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Linear_Audio'
                            ),
                        },
                    step= self.steps,
                    commit= True
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        self.model.train()

    @torch.inference_mode()
    def Inference_Step(self, tokens, token_lengths, speech_prompts, texts, pronunciations, references, start_index= 0, tag_step= False):
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        speech_prompts = speech_prompts.to(self.device, non_blocking=True)

        audio_predictions, *_, durations, f0s, latent_lengths = self.model(
            tokens= tokens,
            token_lengths= token_lengths,
            speech_prompts= speech_prompts,
            ddim_steps= max(self.hp.Diffusion.Max_Step // 10, 100)
            )

        audio_lengths = [
            length * self.hp.Sound.Frame_Shift
            for length in latent_lengths
            ]
        
        audio_predictions = audio_predictions.cpu().numpy()
        f0s = f0s.cpu().numpy()

        durations = [
            torch.arange(duration.size(0)).repeat_interleave(duration.cpu()).numpy()
            for duration in durations
            ]

        files = []
        for index in range(tokens.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (
            audio,
            duration,
            f0,
            token_length,
            latent_length,
            audio_length,
            text,
            pronunciation,
            reference,
            file
            ) in enumerate(zip(
            audio_predictions,
            durations,
            f0s,
            token_lengths.cpu().numpy(),
            latent_lengths.cpu().numpy(),
            audio_lengths,
            texts,
            pronunciations,
            references,
            files
            )):
            title = 'Text: {}    Reference: {}'.format(text if len(text) < 90 else text[:90] + '…', reference)
            new_figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
            ax = plt.subplot2grid((4, 1), (0, 0))
            plt.plot(audio[:audio_length])
            plt.margins(x= 0)
            plt.title(f'Prediction  {title}')
            ax = plt.subplot2grid((4, 1), (1, 0), rowspan= 2)
            plt.plot(duration[:latent_length])
            plt.title('Duration    {}'.format(title))
            plt.margins(x= 0)
            plt.yticks(
                range(len(pronunciation) + 2),
                ['<S>'] + list(pronunciation) + ['<E>'],
                fontsize = 10
                )
            ax = plt.subplot2grid((4, 1), (3, 0), rowspan= 2)
            plt.plot(f0[:latent_length])
            plt.margins(x= 0)
            plt.title('F0    {}'.format(title))
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)
            
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                audio[:audio_length]
                )

    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, token_lengths, speech_prompts, texts, pronunciations, speakers) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(tokens, token_lengths, speech_prompts, texts, pronunciations, speakers, start_index= step * batch_size)

        self.model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])
        self.optimizer.load_state_dict(state_dict['Optimizer'])
        self.scheduler.load_state_dict(state_dict['Scheduler'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps
            }
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        torch.save(state_dict, checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)

    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_path, hp_path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)

        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    argParser.add_argument('-r', '--local-rank', default= 0, type= int)
    args = argParser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl'
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()