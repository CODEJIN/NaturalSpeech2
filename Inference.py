import torch
import logging, yaml, math, sys
from tqdm import tqdm
from typing import List, Optional

from Modules.Modules import NaturalSpeech2
from Datasets import Inference_Dataset as Dataset, Inference_Collater as Collater
from Arg_Parser import Recursive_Parse

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class Inferencer:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,        
        batch_size= 1
        ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.model = NaturalSpeech2(self.hp).to(self.device)
        
        self.Load_Checkpoint(checkpoint_path)
        self.batch_size = batch_size

    def Dataset_Generate(
        self,
        texts: List[str],
        references: List[str],
        ):
        token_dict = yaml.load(open(self.hp.Token_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
        
        return torch.utils.data.DataLoader(
            dataset= Dataset(
                token_dict= token_dict,
                sample_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                texts= texts,
                references= references,
                ),
            shuffle= False,
            collate_fn= Collater(
                token_dict= token_dict,
                speech_prompt_length= self.hp.Train.Inference_in_Train.Speech_Prompt_Length
                ),
            batch_size= self.batch_size,
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
    def Inference_Step(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        speech_prompts: torch.Tensor,
        ddim_steps: Optional[int]= None
        ):
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        speech_prompts = speech_prompts.to(self.device, non_blocking=True)
        
        audios, *_, latent_lengths = self.model(
            tokens= tokens,
            token_lengths= token_lengths,
            speech_prompts= speech_prompts,
            ddim_steps= ddim_steps
            )

        lengths = [
            length * self.hp.Sound.Frame_Shift
            for length in latent_lengths.cpu().numpy()
            ]
        
        audios = [audio[:length] for audio, length in zip(audios.cpu().numpy(), lengths)]
        
        return audios

    def Inference_Epoch(
        self,
        texts: List[str],
        references: List[str],
        ddim_steps: Optional[int]= None,
        use_tqdm: bool= True
        ):
        dataloader = self.Dataset_Generate(
            texts= texts,
            references= references,
            )
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )
        
        audio_list = []
        for tokens, token_lengths, speech_prompts, *_ in dataloader:
            audios = self.Inference_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                speech_prompts= speech_prompts,
                ddim_steps= ddim_steps
                )
            audio_list.extend(audios)
            
        return audio_list
