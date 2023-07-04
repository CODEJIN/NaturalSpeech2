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
from functools import partial

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
        latent_info_dict = yaml.load(open(self.hp.Latent_Info_Path, 'r'), Loader=yaml.Loader)
        self.latent_min = min([x['Min'] for x in latent_info_dict.values()]) / len(latent_info_dict)
        self.latent_max = max([x['Max'] for x in latent_info_dict.values()]) / len(latent_info_dict)
        f0_info_dict = yaml.load(open(self.hp.F0_Info_Path, 'r'), Loader=yaml.Loader)

        train_dataset = Dataset(
            token_dict= token_dict,
            f0_info_dict= f0_info_dict,
            use_between_padding= self.hp.Use_Between_Padding,
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
            use_between_padding= self.hp.Use_Between_Padding,
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
            use_between_padding= self.hp.Use_Between_Padding,
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
        self.model = NaturalSpeech2(
            hyper_parameters= self.hp,
            latent_min= self.latent_min,
            latent_max= self.latent_max
            ).to(self.device)
        self.criterion_dict = {
            'MSE': torch.nn.MSELoss(reduction= 'none').to(self.device),
            'MAE': torch.nn.L1Loss(reduction= 'none').to(self.device),
            'Attention_Binarization': AttentionBinarizationLoss(),
            'Attention_CTC': AttentionCTCLoss(),
            }
        self.mel_func = partial(
            mel_spectrogram,
            n_fft= self.hp.Sound.Frame_Shift * 4,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Frame_Shift,
            win_size= self.hp.Sound.Frame_Shift * 4,
            fmin= 0,
            fmax= None,
            center= False
        )

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
            linear_projections, _, latents_compressed, latents_compressed_slice, noises, epsilons, starts, duration_loss, f0_loss, \
            attention_softs, attention_hards, attention_logprobs, alignments, f0s = self.model(
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
                mel_masks = (~Mask_Generate(
                    lengths= latent_lengths,
                    max_length= mels.size(2)
                    ).to(mels.device)).float()

                loss_dict['Linear'] = (self.criterion_dict['MSE'](
                    linear_projections.float(),
                    latents_compressed,
                    ) * mel_masks.unsqueeze(1)).mean(dim= 1).sum() / mel_masks.sum()
                loss_dict['Data'] = self.criterion_dict['MSE'](
                    starts.float(),
                    latents_compressed_slice,
                    ).mean()
                loss_dict['Diffusion'] = self.criterion_dict['MSE'](
                    epsilons.float(),
                    noises,
                    ).mean()
                loss_dict['Duration'] = (duration_loss.float() * token_masks).sum() / token_masks.sum()
                loss_dict['F0'] = (f0_loss.float() * mel_masks).sum() / mel_masks.sum()
                loss_dict['Attention_Binarization'] = self.criterion_dict['Attention_Binarization'](attention_hards, attention_softs)
                loss_dict['Attention_CTC'] = self.criterion_dict['Attention_CTC'](attention_logprobs, token_lengths, latent_lengths)

        self.optimizer.zero_grad()
        self.scaler.scale(
            loss_dict['Linear'] +
            loss_dict['Data'] / max(starts.max(), 1.0) +
            loss_dict['Diffusion'] +
            loss_dict['Duration'] +
            loss_dict['F0'] +
            loss_dict['Attention_Binarization'] +
            loss_dict['Attention_CTC']
            ).backward()
        
        for name, parameters in self.model.named_parameters():
            if parameters.grad is None:
                continue
            if not name in self.accumulated_grad_dict.keys():
                self.accumulated_grad_dict[name] = 0.0
            self.accumulated_grad_dict[name] += parameters.grad.clone()

        if (self.steps + 1) % self.hp.Train.Accumulated_Gradient_Step == 0:
            for name, parameters in self.model.named_parameters():
                if not name in self.accumulated_grad_dict.keys():
                    continue
                parameters.grad = self.accumulated_grad_dict[name]

            self.scaler.unscale_(self.optimizer)

            if self.hp.Train.Gradient_Norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    parameters= self.model.parameters(),
                    max_norm= self.hp.Train.Gradient_Norm
                    )
        
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.accumulated_grad_dict = {}

        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        self.accumulated_grad_dict = {}
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
            linear_projections, _, latents_compressed, latents_compressed_slice, noises, epsilons, starts, duration_loss, f0_loss, \
            attention_softs, attention_hards, attention_logprobs, alignments, f0s = self.model(
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
                mel_masks = (~Mask_Generate(
                    lengths= latent_lengths,
                    max_length= mels.size(2)
                    ).to(mels.device)).float()

                loss_dict['Linear'] = (self.criterion_dict['MSE'](
                    linear_projections,
                    latents_compressed,
                    ) * mel_masks.unsqueeze(1)).sum() / mel_masks.sum()
                loss_dict['Data'] = self.criterion_dict['MSE'](
                    starts,
                    latents_compressed_slice,
                    ).mean()
                loss_dict['Diffusion'] = self.criterion_dict['MSE'](
                    epsilons,
                    noises,
                    ).mean()
                loss_dict['Duration'] = (duration_loss.float() * token_masks).sum() / token_masks.sum()
                loss_dict['F0'] = (f0_loss.float() * mel_masks).sum() / mel_masks.sum()
                loss_dict['Attention_Binarization'] = self.criterion_dict['Attention_Binarization'](attention_hards, attention_softs)
                loss_dict['Attention_CTC'] = self.criterion_dict['Attention_CTC'](attention_logprobs, token_lengths, latent_lengths)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

        return alignments

    @torch.no_grad()
    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, (tokens, token_lengths, speech_prompts, speech_prompts_for_diffusion, latents, latent_lengths, f0s, mels, attention_priors) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            target_alignments = self.Evaluation_Step(
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
                target_audios = self.model.hificodec(latents[index, None].permute(0, 2, 1).to(self.device)).squeeze(1)
                linear_projections, diffusion_predictions, prediction_alignments, prediction_f0s = self.model.Inference(
                    tokens= tokens[index].unsqueeze(0).to(self.device),
                    token_lengths= token_lengths[index].unsqueeze(0).to(self.device),
                    speech_prompts= speech_prompts[index].unsqueeze(0).to(self.device),
                    ddim_steps= max(self.hp.Diffusion.Max_Step // 10, 100)
                    )
            
            token_length = token_lengths[index].item()
            target_latent_length = target_alignments[index].sum().long().item()
            prediction_latent_length = prediction_alignments[0].sum().long().item()
            target_audio_length = target_latent_length * self.hp.Sound.Frame_Shift
            prediction_audio_length = prediction_latent_length * self.hp.Sound.Frame_Shift

            target_audio = target_audios[0, :target_audio_length].float().clamp(-1.0, 1.0)
            linear_prediction_audio = linear_projections[0, :prediction_audio_length].float().clamp(-1.0, 1.0)
            diffusion_prediction_audio = diffusion_predictions[0, :prediction_audio_length].float().clamp(-1.0, 1.0)

            target_mel = self.mel_func(target_audio[None])[0, :,  :target_latent_length].cpu().numpy()
            linear_prediction_mel = self.mel_func(linear_prediction_audio[None])[0, :,  :prediction_latent_length].cpu().numpy()
            diffusion_prediction_mel = self.mel_func(diffusion_prediction_audio[None])[0, :,  :prediction_latent_length].cpu().numpy()

            target_audio = target_audio.cpu().numpy()
            linear_prediction_audio = linear_prediction_audio.cpu().numpy()
            diffusion_prediction_audio = diffusion_prediction_audio.cpu().numpy()

            target_f0 = f0s[index, :target_latent_length].cpu().numpy() 
            prediction_f0 = prediction_f0s[0, :prediction_latent_length].cpu().numpy()

            target_alignment = target_alignments[index, :token_length, :target_latent_length].cpu().numpy()
            prediction_alignment = prediction_alignments[0, :token_length, :prediction_latent_length].cpu().numpy()

            image_dict = {
                'Feature/Target': (target_mel, None, 'auto', None, None, None),
                'Feature/Linear': (linear_prediction_mel, None, 'auto', None, None, None),
                'Feature/Diffusion': (diffusion_prediction_mel, None, 'auto', None, None, None),
                'Alignment/Target': (target_alignment, None, 'auto', None, None, None),
                'Alignment/Prediction': (prediction_alignment, None, 'auto', None, None, None),
                'F0/Target': (target_f0, None, 'auto', None, None, None),
                'F0/Prediction': (prediction_f0, None, 'auto', None, None, None),
                }
            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Linear': (linear_prediction_audio, self.hp.Sound.Sample_Rate),
                'Audio/Diffusion': (diffusion_prediction_audio, self.hp.Sound.Sample_Rate),
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
                        'Evaluation.Feature.Target': wandb.Image(target_mel),
                        'Evaluation.Feature.Linear': wandb.Image(linear_prediction_mel),
                        'Evaluation.Feature.Diffusion': wandb.Image(diffusion_prediction_mel),
                        'Evaluation.F0': wandb.plot.line_series(
                            xs= np.arange(max(target_latent_length, prediction_latent_length)),
                            ys= [target_f0, prediction_f0],
                            keys= ['Target', 'Prediction'],
                            title= 'F0',
                            xname= 'Mel_t'
                            ),
                        'Evaluation.Alignment.Target': wandb.Image(target_alignment),
                        'Evaluation.Alignment.Prediction': wandb.Image(prediction_alignment),
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Target_Audio'
                            ),
                        'Evaluation.Audio.Linear': wandb.Audio(
                            linear_prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Linear_Audio'
                            ),
                        'Evaluation.Audio.Diffusion': wandb.Audio(
                            diffusion_prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Diffusion_Audio'
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

        linear_predictions, diffusion_predictions, alignments, f0s = self.model.Inference(
            tokens= tokens,
            token_lengths= token_lengths,
            speech_prompts= speech_prompts,
            ddim_steps= max(self.hp.Diffusion.Max_Step // 10, 100)
            )

        latent_lengths = [length for length in alignments.sum(dim= [1, 2]).long().cpu().numpy()]
        audio_lengths = [
            length * self.hp.Sound.Frame_Shift
            for length in latent_lengths
            ]
        
        linear_audio_predictions = [
            audio[:length]
            for audio, length in zip(linear_predictions.cpu().numpy(), audio_lengths)
            ]
        diffusion_audio_predictions = [
            audio[:length]
            for audio, length in zip(diffusion_predictions.cpu().numpy(), audio_lengths)
            ]
        
        linear_mel_predictions = [
            mel[:, :length]
            for mel, length in zip(self.mel_func(linear_predictions).cpu().numpy(), latent_lengths)
            ]
        diffusion_mel_predictions = [
            mel[:, :length]
            for mel, length in zip(self.mel_func(diffusion_predictions).cpu().numpy(), latent_lengths)
            ]
        
        alignments = [
            alignment[:token_length, :mel_length]
            for alignment, token_length, mel_length in zip(alignments.cpu().numpy(), token_lengths, latent_lengths)
            ]        
        f0s = [
            f0[:length]
            for f0, length in zip(f0s.cpu().numpy(), latent_lengths)
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
            linear_mel,
            diffusion_mel,
            alignment,
            f0,
            linear_audio,
            diffusion_audio,
            text,
            pronunciation,
            reference,
            file
            ) in enumerate(zip(
            linear_mel_predictions,
            diffusion_mel_predictions,
            alignments,
            f0s,
            linear_audio_predictions,
            diffusion_audio_predictions,
            texts,
            pronunciations,
            references,
            files
            )):
            title = 'Text: {}    Reference: {}'.format(text if len(text) < 90 else text[:90] + '…', reference)
            new_figure = plt.figure(figsize=(20, 5 * 5), dpi=100)
            
            ax = plt.subplot2grid((5, 1), (0, 0))
            plt.imshow(linear_mel, aspect= 'auto', origin= 'lower')
            plt.margins(x= 0)
            plt.title(f'Linear prediction  {title}')            
            ax = plt.subplot2grid((5, 1), (1, 0))
            plt.imshow(diffusion_mel, aspect= 'auto', origin= 'lower')
            plt.margins(x= 0)
            plt.title(f'Diffusion prediction  {title}')            
            ax = plt.subplot2grid((5, 1), (2, 0), rowspan= 2)
            plt.imshow(alignment, aspect= 'auto', origin= 'lower')
            plt.margins(x= 0)
            plt.yticks(
                range(len(pronunciation) + 2),
                ['<S>'] + list(pronunciation) + ['<E>'],
                fontsize = 10
                )
            plt.title(f'Alignment  {title}')
            ax = plt.subplot2grid((5, 1), (4, 0))
            plt.plot(f0)
            plt.margins(x= 0)
            plt.title('F0    {}'.format(title))
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)
            
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.Linear.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                linear_audio
                )
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.Diffusion.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                diffusion_audio
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-s', '--steps', default= 0, type= int)
    parser.add_argument('-r', '--local-rank', default= 0, type= int)
    args = parser.parse_args()

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
    trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    trainer.Train()