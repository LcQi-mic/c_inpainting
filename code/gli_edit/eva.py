import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn, autograd  

import torch.nn.functional as F
import numpy as np



from model import Editor
from dataloader import get_loader

from monai.metrics import DiceMetric, MAEMetric, PSNRMetric, DiceHelper
import clip


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

class Evaluator:
    def __init__(self, 
                args=None):
        
        self.args = args
        self.global_step = 0
        
        self.trainer_initialized = False
        self.device = 'cuda'

    def initial_trainer(self):
        self.trainer_initialized = False
        self.config_dataset()
        self.init_model()
        self.init_metric()
        
        self.trainer_initialized = True
        
    def config_dataset(self):
        self.test_dataloader = get_loader(self.args.test_json,
                                1,
                                int(self.args.workers),
                                phase='train')
    
    def init_model(self):
        self.G = Editor(     
                in_c=self.args.in_c,
                filters=self.args.filters,
                style_dim=self.args.style_dim,
                out_c=self.args.out_c       
            )
        
        self.clip, _ = clip.load("ViT-B/16")
        self.sigmoid = torch.nn.Sigmoid()

        pytorch_encoder_params = sum(p.numel() for p in self.G.parameters() if p.requires_grad)
        print("Transformer parameters count", pytorch_encoder_params)
        
    def init_metric(self):
        self.dice = DiceHelper(include_background=True, sigmoid=True)
        self.psnr = PSNRMetric()
        self.mae = MAEMetric()
        
    def validate(self):
        self.G.eval()
        
        delta_small = []
        delta_middle = []
        delta_large = []

        for idx, batch_data in enumerate(self.test_dataloader):
            gli_0, gli_1, prompt_0, prompt_1, token = batch_data['gli_0'], batch_data['gli_1'], batch_data['prompt_0'], batch_data['prompt_1'], batch_data['token']
            # w = self.clip.encode_text(token.to(torch.int32).squeeze(1)).to(torch.float32)
            
            delta = self.get_delta(prompt_0, prompt_1)
            
            if 0.15 < delta < 0.2:
                print(batch_data['path'])
            assert 1 == 2
            # with torch.no_grad():
            #     y_hat = self.G(gli_0, w)
                
            #     metirc_dict = self.calc_metric(y=gli_1, y_hat=y_hat)
                
            #     if delta > 0 and delta < 0.3:
            #         delta_small.append(metirc_dict['dice'])
            #     elif delta >= 0.3 and delta < 0.6:
            #         delta_middle.append(metirc_dict['dice'])
            #     elif delta >= 0.6 and delta < 0.99:
            #         delta_large.append(metirc_dict['dice'])

        return [delta_small / len(delta_small), delta_middle / len(delta_middle), delta_large / len(delta_large)]
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        self.G.load_state_dict(new_state_dict, strict=False)
        print("=> loaded model checkpoint")
        
    def load_G(self):
        encoder = torch.load(self.args.G, map_location=torch.device('cpu'))
        return encoder

    def calc_metric(self, y, y_hat):
        metric_dict = {}
        
        loss_dice = self.dice(y_hat, y)
        metric_dict['dice'] = float(loss_dice)
        
        return metric_dict
    
    def get_delta(self, prompt_0, prompt_1):
        prompt_0 = prompt_0
        prompt_1 = prompt_1
        
        size_0 = prompt_0[0][-1] + prompt_0[1][-1] + prompt_0[2][-1]
        size_1 = prompt_1[0][-1] + prompt_1[1][-1] + prompt_1[2][-1]
        
        if size_0 == 0:
            delta = 1
        else:
            delta = abs(size_1 - size_0) / max(size_0, size_1)
            
        return delta
    


