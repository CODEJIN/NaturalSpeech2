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

from Modules.Modules import HierSpeech, Mask_Generate
from Modules.Discriminator import Discriminator, R1_Regulator, Feature_Map_Loss, Generator_Loss, Discriminator_Loss
from Modules.Flow import Flow_KL_Loss

from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
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
                wandb.watch(self.model_dict['HierSpeech'])

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
        ge2e_dict = pickle.load(open(self.hp.GE2E_Path, 'rb'))
        f0_info_dict = yaml.load(open(self.hp.F0_Info_Path, 'r'), Loader=yaml.Loader)

        train_dataset = Dataset(
            token_dict= token_dict,
            f0_info_dict= f0_info_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            feature_length_min= max(self.hp.Train.Train_Pattern.Feature_Length.Min, self.hp.Train.Segment_Size),
            feature_length_max= self.hp.Train.Train_Pattern.Feature_Length.Max,
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
            feature_length_min= max(self.hp.Train.Train_Pattern.Feature_Length.Min, self.hp.Train.Segment_Size),
            feature_length_max= self.hp.Train.Eval_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Eval_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Eval_Pattern.Text_Length.Max,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            ge2e_dict= ge2e_dict,
            texts= self.hp.Train.Inference_in_Train.Text,
            speakers= self.hp.Train.Inference_in_Train.Speaker,
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(
            token_dict= token_dict,
            hop_size= self.hp.Sound.Frame_Shift
            )
        inference_collater = Inference_Collater(
            token_dict= token_dict
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
        self.model_dict = {
            'HierSpeech': HierSpeech(self.hp).to(self.device),
            'Discriminator': Discriminator(
                use_stft_discriminator= self.hp.Discriminator.Use_STFT,
                period_list= self.hp.Discriminator.Period,
                stft_n_fft_list= self.hp.Discriminator.STFT_N_FFT,
                scale_pool_kernel_size_list= self.hp.Discriminator.Scale_Pool_Kernel_Size,
                ).to(self.device),
            }
        self.criterion_dict = {
            'MSE': torch.nn.MSELoss(reduce= None).to(self.device),
            'MAE': torch.nn.L1Loss(reduce= None).to(self.device),
            'TokenCTC': torch.nn.CTCLoss(
                blank= self.hp.Tokens,  # == Token length
                zero_infinity=True
                ),
            'R1': R1_Regulator()
            }
        self.optimizer_dict = {
            'HierSpeech': torch.optim.AdamW(
                params= self.model_dict['HierSpeech'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon
                ),
            'Discriminator': torch.optim.AdamW(
                params= self.model_dict['Discriminator'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon
                ),
            }
        self.scheduler_dict = {
            'HierSpeech': torch.optim.lr_scheduler.ExponentialLR(
                optimizer= self.optimizer_dict['HierSpeech'],
                gamma= self.hp.Train.Learning_Rate.Decay,
                last_epoch= -1
                ),
            'Discriminator': torch.optim.lr_scheduler.ExponentialLR(
                optimizer= self.optimizer_dict['Discriminator'],
                gamma= self.hp.Train.Learning_Rate.Decay,
                last_epoch= -1
                ),
            }

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        # if self.gpu_id == 0:
        #     logging.info(self.model_dict['HierSpeech'])

    def Train_Step(self, tokens, token_lengths, ge2es, features, feature_lengths, f0s, audios):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        ge2es = ge2es.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)
        feature_lengths = feature_lengths.to(self.device, non_blocking=True)
        f0s = f0s.to(self.device, non_blocking=True)
        audios = audios.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            audio_predictions_slice, audios_slice, mel_predictions_slice, mels_slice, \
            audio_predictions_forward_slice, audios_forward_slice, \
            encoding_means, encoding_log_stds, linguistic_flows, linguistic_log_stds, \
            linguistic_means, linguistic_log_stds, linguistic_samples_forward, encoding_log_stds, \
            linguistic_means, linguistic_log_stds, acoustic_flows, acoustic_log_stds, \
            acoustic_means, acoustic_log_stds, acoustic_samples_forward, linguistic_log_stds, \
            duration_losses, token_predictions, f0_predictions, _ = self.model_dict['HierSpeech'](
                tokens= tokens,
                token_lengths= token_lengths,
                ge2es= ge2es,
                features= features,
                feature_lengths= feature_lengths,
                f0s= f0s,
                audios= audios
                )
            
            audios_slice.requires_grad_()
            audios_forward_slice.requires_grad_()
            discriminations_list_for_real, _ = self.model_dict['Discriminator'](audios_slice,)
            discriminations_list_for_fake, _ = self.model_dict['Discriminator'](audio_predictions_slice.detach())
            discriminations_forward_list_for_real, _ = self.model_dict['Discriminator'](audios_forward_slice,)
            discriminations_forward_list_for_fake, _ = self.model_dict['Discriminator'](audio_predictions_forward_slice.detach())
            with torch.cuda.amp.autocast(enabled= False):
                loss_dict['Discrimination'] = Discriminator_Loss(discriminations_list_for_real, discriminations_list_for_fake)
                loss_dict['Discrimination_Forwad'] = Discriminator_Loss(discriminations_forward_list_for_real, discriminations_forward_list_for_fake)
                loss_dict['R1'] = self.criterion_dict['R1'](discriminations_list_for_real, audios_slice)
                loss_dict['R1_Forward'] = self.criterion_dict['R1'](discriminations_forward_list_for_real, audios_forward_slice)

        self.optimizer_dict['Discriminator'].zero_grad()
        self.scaler.scale(
            loss_dict['Discrimination'] +
            loss_dict['Discrimination_Forwad'] * self.hp.Train.Learning_Rate.Lambda.GAN_Forward +
            loss_dict['R1'] +
            loss_dict['R1_Forward']
            ).backward()
        self.scaler.unscale_(self.optimizer_dict['Discriminator'])
        if self.hp.Train.Gradient_Norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_dict['Discriminator'].parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer_dict['Discriminator'])
        self.scaler.update()

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            discriminations_list_for_real, feature_maps_list_for_real = self.model_dict['Discriminator'](audios_slice,)
            discriminations_list_for_fake, feature_maps_list_for_fake = self.model_dict['Discriminator'](audio_predictions_slice)            
            discriminations_forward_list_for_fake, _ = self.model_dict['Discriminator'](audio_predictions_forward_slice)
            with torch.cuda.amp.autocast(enabled= False):
                feature_masks = Mask_Generate(
                    lengths= feature_lengths,
                    max_length= features.size(2)
                    ).to(features.device)
                        
                loss_dict['STFT'] = self.criterion_dict['MAE'](
                    mel_predictions_slice,
                    mels_slice
                    ).mean()
                loss_dict['F0'] = (self.criterion_dict['MSE'](
                    f0_predictions.float(),
                    f0s
                    ) * ~feature_masks.unsqueeze(1)).mean()
                loss_dict['Duration'] = duration_losses.float().sum()
                loss_dict['TokenCTC'] = self.criterion_dict['TokenCTC'](
                    log_probs= token_predictions.permute(2, 0, 1),  # [Feature_t, Batch, Token_n]
                    targets= tokens,
                    input_lengths= feature_lengths,
                    target_lengths= token_lengths
                    )
                loss_dict['Encoding_KLD'] = Flow_KL_Loss(
                    encoding_means= encoding_means,
                    encoding_log_stds= encoding_log_stds,
                    flows= linguistic_flows,
                    flow_log_stds= linguistic_log_stds,
                    masks= ~feature_masks.unsqueeze(1)
                    )
                loss_dict['Encoding_KLD_Forward'] = Flow_KL_Loss(
                    encoding_means= linguistic_means,
                    encoding_log_stds= linguistic_log_stds,
                    flows= linguistic_samples_forward,
                    flow_log_stds= encoding_log_stds,
                    masks= ~feature_masks.unsqueeze(1)
                    )
                loss_dict['Linguistic_KLD'] = Flow_KL_Loss(
                    encoding_means= linguistic_means,
                    encoding_log_stds= linguistic_log_stds,
                    flows= acoustic_flows,
                    flow_log_stds= acoustic_log_stds,
                    masks= ~feature_masks.unsqueeze(1)
                    )
                loss_dict['Linguistic_KLD_Forward'] = Flow_KL_Loss(
                    encoding_means= acoustic_means,
                    encoding_log_stds= acoustic_log_stds,
                    flows= acoustic_samples_forward,
                    flow_log_stds= linguistic_log_stds,
                    masks= ~feature_masks.unsqueeze(1)
                    )

                loss_dict['Adversarial'] = Generator_Loss(discriminations_list_for_fake)
                loss_dict['Adversarial_Forward'] = Generator_Loss(discriminations_forward_list_for_fake)
                loss_dict['Feature_Map'] = Feature_Map_Loss(feature_maps_list_for_real, feature_maps_list_for_fake)

        self.optimizer_dict['HierSpeech'].zero_grad()
        self.scaler.scale(
            loss_dict['STFT'] * self.hp.Train.Learning_Rate.Lambda.STFT +
            loss_dict['F0'] * self.hp.Train.Learning_Rate.Lambda.F0 +
            loss_dict['Duration'] +
            loss_dict['TokenCTC'] * self.hp.Train.Learning_Rate.Lambda.Token_CTC +
            loss_dict['Encoding_KLD'] +
            loss_dict['Encoding_KLD_Forward'] * self.hp.Train.Learning_Rate.Lambda.KLD_Forward +
            loss_dict['Linguistic_KLD'] +
            loss_dict['Linguistic_KLD_Forward'] * self.hp.Train.Learning_Rate.Lambda.KLD_Forward +
            loss_dict['Adversarial'] +
            loss_dict['Adversarial_Forward'] * self.hp.Train.Learning_Rate.Lambda.GAN_Forward +
            loss_dict['Feature_Map'] * self.hp.Train.Learning_Rate.Lambda.Feature_Map
            ).backward()

        self.scaler.unscale_(self.optimizer_dict['HierSpeech'])

        if self.hp.Train.Gradient_Norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_dict['HierSpeech'].parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer_dict['HierSpeech'])
        self.scaler.update()

        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for tokens, token_lengths, ge2es, features, feature_lengths, f0s, audios in self.dataloader_dict['Train']:
            self.Train_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                ge2es= ge2es,
                features= features,
                feature_lengths= feature_lengths,
                f0s= f0s,
                audios= audios
                )

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler_dict['HierSpeech'].get_last_lr()[0]
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

        self.scheduler_dict['Discriminator'].step()
        self.scheduler_dict['HierSpeech'].step()

    def Evaluation_Step(self, tokens, token_lengths, ge2es, features, feature_lengths, f0s, audios):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        ge2es = ge2es.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)
        feature_lengths = feature_lengths.to(self.device, non_blocking=True)
        f0s = f0s.to(self.device, non_blocking=True)
        audios = audios.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            audio_predictions_slice, audios_slice, mel_predictions_slice, mels_slice, \
            audio_predictions_forward_slice, audios_forward_slice, \
            encoding_means, encoding_log_stds, linguistic_flows, linguistic_log_stds, \
            linguistic_means, linguistic_log_stds, linguistic_samples_forward, encoding_log_stds, \
            linguistic_means, linguistic_log_stds, acoustic_flows, acoustic_log_stds, \
            acoustic_means, acoustic_log_stds, acoustic_samples_forward, linguistic_log_stds, \
            duration_losses, token_predictions, f0_predictions, _ = self.model_dict['HierSpeech'](
                tokens= tokens,
                token_lengths= token_lengths,
                ge2es= ge2es,
                features= features,
                feature_lengths= feature_lengths,
                f0s= f0s,
                audios= audios
                )

            audios_slice.requires_grad_() # to calculate gradient penalty.
            audios_forward_slice.requires_grad_()
            discriminations_list_for_real, feature_maps_list_for_real = self.model_dict['Discriminator'](audios_slice)
            discriminations_list_for_fake, feature_maps_list_for_fake = self.model_dict['Discriminator'](audio_predictions_slice)
            discriminations_forward_list_for_real, _ = self.model_dict['Discriminator'](audios_forward_slice,)
            discriminations_forward_list_for_fake, _ = self.model_dict['Discriminator'](audio_predictions_forward_slice.detach())
            with torch.cuda.amp.autocast(enabled= False):
                feature_masks = Mask_Generate(
                    lengths= feature_lengths,
                    max_length= features.size(2)
                    ).to(features.device)

                loss_dict['STFT'] = self.criterion_dict['MAE'](
                    mel_predictions_slice,
                    mels_slice
                    ).mean()
                loss_dict['F0'] = (self.criterion_dict['MSE'](
                    f0_predictions,
                    f0s
                    ) * ~feature_masks.unsqueeze(1)).mean()
                loss_dict['Duration'] = duration_losses.float().sum()
                loss_dict['TokenCTC'] = self.criterion_dict['TokenCTC'](
                    log_probs= token_predictions.permute(2, 0, 1),  # [Feature_t, Batch, Token_n]
                    targets= tokens,
                    input_lengths= feature_lengths,
                    target_lengths= token_lengths
                    )
                loss_dict['Encoding_KLD'] = Flow_KL_Loss(
                    encoding_means= encoding_means,
                    encoding_log_stds= encoding_log_stds,
                    flows= linguistic_flows,
                    flow_log_stds= linguistic_log_stds,
                    masks= ~feature_masks.unsqueeze(1)
                    )
                loss_dict['Encoding_KLD_Forward'] = Flow_KL_Loss(
                    encoding_means= linguistic_means,
                    encoding_log_stds= linguistic_log_stds,
                    flows= linguistic_samples_forward,
                    flow_log_stds= encoding_log_stds,
                    masks= ~feature_masks.unsqueeze(1)
                    )
                loss_dict['Linguistic_KLD'] = Flow_KL_Loss(
                    encoding_means= linguistic_means,
                    encoding_log_stds= linguistic_log_stds,
                    flows= acoustic_flows,
                    flow_log_stds= acoustic_log_stds,
                    masks= ~feature_masks.unsqueeze(1)
                    )
                loss_dict['Linguistic_KLD_Forward'] = Flow_KL_Loss(
                    encoding_means= acoustic_means,
                    encoding_log_stds= acoustic_log_stds,
                    flows= acoustic_samples_forward,
                    flow_log_stds= linguistic_log_stds,
                    masks= ~feature_masks.unsqueeze(1)
                    )
                
                loss_dict['Discrimination'] = Discriminator_Loss(discriminations_list_for_real, discriminations_list_for_fake)
                loss_dict['Discrimination_Forwad'] = Discriminator_Loss(discriminations_forward_list_for_real, discriminations_forward_list_for_fake)
                loss_dict['R1'] = self.criterion_dict['R1'](discriminations_list_for_real, audios_slice)
                loss_dict['R1_Forward'] = self.criterion_dict['R1'](discriminations_forward_list_for_real, audios_forward_slice)
                loss_dict['Adversarial'] = Generator_Loss(discriminations_list_for_fake)
                loss_dict['Adversarial_Forward'] = Generator_Loss(discriminations_forward_list_for_fake)
                loss_dict['Feature_Map'] = Feature_Map_Loss(feature_maps_list_for_real, feature_maps_list_for_fake)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        for model in self.model_dict.values():
            model.eval()

        for step, (tokens, token_lengths, ge2es, features, feature_lengths, f0s, audios) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            self.Evaluation_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                ge2es= ge2es,
                features= features,
                feature_lengths= feature_lengths,
                f0s= f0s,
                audios= audios
                )

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['HierSpeech'], 'HierSpeech', self.steps, delete_keywords=[])
            self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['Discriminator'], 'Discriminator', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, tokens.size(0))

            with torch.inference_mode():
                prediction_audios, *_, f0_predictions, alignments = self.model_dict['HierSpeech'](
                    tokens= tokens[index].unsqueeze(0).to(self.device),
                    token_lengths= token_lengths[index].unsqueeze(0).to(self.device),
                    ge2es= ge2es[index].unsqueeze(0).to(self.device),
                    )
            
            token_length = token_lengths[index]
            target_feature_length = feature_lengths[index]
            prediction_feature_length = alignments[0].sum().long()
            target_audio_length = target_feature_length * self.hp.Sound.Frame_Shift
            prediction_audio_length = prediction_feature_length * self.hp.Sound.Frame_Shift
            
            target_audio = audios[index, :target_audio_length]
            prediction_audio = prediction_audios[0, :prediction_audio_length]

            target_feature = mel_spectrogram(
                target_audio.unsqueeze(0),
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Length,
                fmin= 0,
                fmax= None
                ).squeeze(0).cpu().numpy()
            
            prediction_feature = mel_spectrogram(
                prediction_audio.unsqueeze(0),
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Length,
                fmin= 0,
                fmax= None
                ).squeeze(0).cpu().numpy()
            
            target_audio = target_audio.cpu().numpy()
            prediction_audio = prediction_audio.cpu().numpy()

            target_f0 = f0s[index, :target_feature_length].cpu().numpy() 
            prediction_f0 = f0_predictions[0, :prediction_feature_length].cpu().numpy()

            prediction_alignment = alignments[0, :prediction_feature_length, :token_length].cpu().numpy()

            image_dict = {
                'Feature/Target': (target_feature, None, 'auto', None, None, None),
                'Feature/Prediction': (prediction_feature, None, 'auto', None, None, None),
                'Duration/Prediction': (prediction_alignment.T, None, 'auto', None, None, None),
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
                            xname= 'Feature_t'
                            ),
                        'Evaluation.Alignment': wandb.Image(prediction_alignment.T),
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

        for model in self.model_dict.values():
            model.train()

    @torch.inference_mode()
    def Inference_Step(self, tokens, token_lengths, ge2es, texts, pronunciations, speakers, start_index= 0, tag_step= False):
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        ge2es = ge2es.to(self.device, non_blocking=True)

        audio_predictions, *_, f0s, alignments = self.model_dict['HierSpeech'](
            tokens= tokens,
            token_lengths= token_lengths,
            ge2es= ge2es
            )

        feature_lengths = alignments.sum(dim= [1, 2]).long()
        audio_lengths = [
            length * self.hp.Sound.Frame_Shift
            for length in feature_lengths
            ]

        feature_predictions = mel_spectrogram(
            audio_predictions,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Frame_Shift,
            win_size= self.hp.Sound.Frame_Length,
            fmin= 0,
            fmax= None
            ).cpu().numpy()
        audio_predictions = audio_predictions.cpu().numpy()
        f0s = f0s.cpu().numpy()
        alignments = alignments.cpu().numpy()

        files = []
        for index in range(tokens.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (
            feature,
            audio,
            alignment,
            f0,
            token_length,
            feature_length,
            audio_length,
            text,
            pronunciation,
            speaker,
            file
            ) in enumerate(zip(
            feature_predictions,
            audio_predictions,
            alignments,
            f0s,
            token_lengths,
            feature_lengths,
            audio_lengths,
            texts,
            pronunciations,
            speakers,
            files
            )):
            title = 'Text: {}    Speaker: {}'.format(text if len(text) < 90 else text[:90] + '…', speaker)
            new_figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
            ax = plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(feature[:, :feature_length], aspect='auto', origin='lower')
            plt.title(f'Prediction  {title}')
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((4, 1), (1, 0), rowspan= 2)
            plt.imshow(alignment[:feature_length, :token_length].T, aspect='auto', origin='lower')
            plt.title('Alignment    {}'.format(title))
            plt.yticks(
                range(len(pronunciation) + 2),
                ['<S>'] + list(pronunciation) + ['<E>'],
                fontsize = 10
                )
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((4, 1), (3, 0), rowspan= 2)
            plt.plot(f0[:feature_length])
            plt.margins(x= 0)
            plt.title('F0    {}'.format(title))
            plt.colorbar(ax= ax)
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

        for model in self.model_dict.values():
            model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, token_lengths, ge2es, texts, pronunciations, speakers) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(tokens, token_lengths, ge2es, texts, pronunciations, speakers, start_index= step * batch_size)

        for model in self.model_dict.values():
            model.train()

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
        self.model_dict['HierSpeech'].load_state_dict(state_dict['Model']['HierSpeech'])
        self.model_dict['Discriminator'].load_state_dict(state_dict['Model']['Discriminator'])
        self.optimizer_dict['HierSpeech'].load_state_dict(state_dict['Optimizer']['HierSpeech'])
        self.optimizer_dict['Discriminator'].load_state_dict(state_dict['Optimizer']['Discriminator'])
        self.scheduler_dict['HierSpeech'].load_state_dict(state_dict['Scheduler']['HierSpeech'])
        self.scheduler_dict['Discriminator'].load_state_dict(state_dict['Scheduler']['Discriminator'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': {
                'HierSpeech': self.model_dict['HierSpeech'].state_dict(),
                'Discriminator': self.model_dict['Discriminator'].state_dict(),
                },
            'Optimizer': {
                'HierSpeech': self.optimizer_dict['HierSpeech'].state_dict(),
                'Discriminator': self.optimizer_dict['Discriminator'].state_dict(),
                },
            'Scheduler': {
                'HierSpeech': self.scheduler_dict['HierSpeech'].state_dict(),
                'Discriminator': self.scheduler_dict['Discriminator'].state_dict(),
                },
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
            self.model_dict = {
                key: apply_gradient_allreduce(model)
                for key, model in self.model_dict.items()
                }

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