import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import logging, yaml, math, sys, os, pickle, argparse
from random import sample
from tqdm import tqdm
from typing import List, Optional, Tuple

from Modules.Modules import NaturalSpeech2
from Datasets import Latent_Stack
from Arg_Parser import Recursive_Parse

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path: str,
        metadata_file: str,
        latent_length_min: int,
        latent_length_max: int,
        num_speaker: int
        ):
        super().__init__()
        self.pattern_path = pattern_path

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        
        self.patterns = []
        for patterns in sample(sorted(metadata_dict['File_List_by_Speaker_Dict'].values()), num_speaker):            
            self.patterns.extend(sorted(sample(patterns, min(50, len(patterns)))))

        self.patterns = [
            x for x in self.patterns
            if all([
                metadata_dict['Latent_Length_Dict'][x] >= latent_length_min,
                metadata_dict['Latent_Length_Dict'][x] <= latent_length_max
                ])
            ]
    
    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')        
        pattern_dict = pickle.load(open(path, 'rb'))
        
        return pattern_dict['Latent'], pattern_dict['Speaker']

    def __len__(self):
        return len(self.patterns)    

class Collater:
    def __init__(self,
        speech_prompt_length: int
        ):
        self.speech_prompt_length = speech_prompt_length
         
    def __call__(self, batch):
        speech_prompt_latents, speakers = zip(*batch)
        speech_prompt_latent_lengths = np.array([latent.shape[1] for latent in speech_prompt_latents])

        speech_prompt_length = min(self.speech_prompt_length, speech_prompt_latent_lengths.min())

        speech_prompts = []
        for latent in speech_prompt_latents:
            offset = np.random.randint(0, latent.shape[1] - speech_prompt_length + 1)
            speech_prompts.append(latent[:, offset:offset + speech_prompt_length])
        
        speech_prompts = Latent_Stack(speech_prompts)        
        speech_prompts = torch.LongTensor(speech_prompts)
        
        return speech_prompts, speakers

class Checker:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,
        num_speaker: int
        ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        latent_info_dict = yaml.load(open(self.hp.Latent_Info_Path, 'r'), Loader=yaml.Loader)
        self.latent_min = min([x['Min'] for x in latent_info_dict.values()])
        self.latent_max = max([x['Max'] for x in latent_info_dict.values()])

        self.model = NaturalSpeech2(
            hyper_parameters= self.hp,
            latent_min= self.latent_min,
            latent_max= self.latent_max
            ).to(self.device)
        
        self.Load_Checkpoint(checkpoint_path)
        print(f'Pre attention query min: {self.model.diffusion.network.pre_attention_query.min()}')
        print(f'Pre attention query max: {self.model.diffusion.network.pre_attention_query.max()}')

        self.Dataset_Generate(num_speaker= num_speaker)

    def Dataset_Generate(
        self,
        num_speaker: int
        ):
        dataset = Dataset(
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            latent_length_min= max(self.hp.Train.Train_Pattern.Feature_Length.Min, self.hp.Train.Segment_Size),
            latent_length_max= self.hp.Train.Train_Pattern.Feature_Length.Max,
            num_speaker= num_speaker
            )
        collater = Collater(
            speech_prompt_length= self.hp.Train.Inference_in_Train.Speech_Prompt_Length
            )
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset= dataset,
            shuffle= False,
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= 0,
            pin_memory= True
            )

    def Load_Checkpoint(self, path: str):
        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])
        self.steps = state_dict['Steps']

        self.model.eval()

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    @torch.inference_mode()
    def Clustering_Step(
        self,
        speech_prompts: torch.Tensor,        
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        speech_prompts = speech_prompts.to(self.device, non_blocking=True)        
        speech_prompts = self.model.hificodec.quantizer.embed(speech_prompts.permute(0, 2, 1)) # type: ignore
        
        speech_prompts = self.model.speech_prompter(speech_prompts)
        diffusion_prompts = self.model.diffusion.network.pre_attention(
            queries= self.model.diffusion.network.pre_attention_query.expand(speech_prompts.size(0), -1, -1),
            keys= speech_prompts,
            values= speech_prompts
            )   # [Batch, Diffusion_d, Token_n]
        
        return speech_prompts, diffusion_prompts

    @torch.inference_mode()
    def Clustering(self, output_path: Optional[str]= None):
        output_path = output_path or 'clustering.png'

        speech_prompt_list, diffusion_prompt_list, speaker_list = [], [], []
        for step, (speech_prompts, speakers) in tqdm(
            enumerate(self.dataloader, 1),
            desc='[Clustering]',
            total= math.ceil(len(self.dataloader.dataset) / self.hp.Train.Batch_Size)
            ):
            speech_prompts, diffusion_prompts = self.Clustering_Step(speech_prompts= speech_prompts)
            speech_prompt_list.extend(speech_prompts.mean(dim= 2).cpu().numpy())
            diffusion_prompt_list.extend(diffusion_prompts.mean(dim= 2).cpu().numpy())
            speaker_list.extend(speakers)

        speech_prompts = np.stack(speech_prompt_list, axis= 0)
        diffusion_prompts = np.stack(diffusion_prompt_list, axis= 0)

        print(f'Speech prompt min: {speech_prompts.min()}')
        print(f'Speech prompt max: {speech_prompts.max()}')
        print(f'Diffusion prompt min: {diffusion_prompts.min()}')
        print(f'Diffusion prompt max: {diffusion_prompts.max()}')

        speech_prompt_pca = PCA(n_components= 2)
        speech_prompt_pca.fit(speech_prompts)
        speech_prompt_pca = speech_prompt_pca.transform(speech_prompts)

        diffusion_prompt_pca = PCA(n_components= 2)
        diffusion_prompt_pca.fit(diffusion_prompts)
        diffusion_prompt_pca = diffusion_prompt_pca.transform(diffusion_prompts)


        df = pd.concat([
            pd.DataFrame(
                data= speech_prompt_pca,
                columns= ['SP_1', 'SP_2']
                ),
            pd.DataFrame(
                data= diffusion_prompt_pca,
                columns= ['DP_1', 'DP_2']
                )
            ], axis= 1)        
        df['Speaker'] = speaker_list

        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x= 'SP_1', y= 'SP_2', hue= 'Speaker', data= df)
        plt.title('Speech Prompt')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x= 'DP_1', y= 'DP_2', hue= 'Speaker', data= df)
        plt.title('Diffusion Prompt')

        plt.tight_layout()
        plt.savefig(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-c', '--checkpoint_path', required= True, type= str)
    parser.add_argument('-o', '--output_path', required= False, default= 'clustering.png', type= str)
    parser.add_argument('-n', '--num_speaker', required= False, default= 20, type= int)
    args = parser.parse_args()

    checker = Checker(
        hp_path= args.hyper_parameters,
        checkpoint_path= args.checkpoint_path,
        num_speaker= args.num_speaker
        )
    checker.Clustering(
        output_path= args.output_path
        )
    
# python Cluster_Check.py -hp Hyper_Parameters.yaml -c ./results/VCTK_230805/Checkpoint/S_157173.pt -o ./S_157173.png -n 20