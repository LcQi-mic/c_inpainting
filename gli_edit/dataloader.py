import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai import transforms
import SimpleITK as sitk
import json
import os
import clip


def datafold_read(datalist):
    with open(datalist) as f:
        json_data = json.load(f)

    return json_data

def trans_brats_label(x):
        mask_WT = x.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 3] = 1

        mask_TC = x.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 3] = 1

        mask_ET = x.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 3] = 1
        
        mask = np.stack([mask_WT, mask_TC, mask_ET], axis=0)
        return mask


class BraTsDataset(Dataset):
    def __init__(self, data_list, args=None, phase='train'):
        super(BraTsDataset, self).__init__()
        self.data_list = data_list['train']

        self.train_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys=["mri_0", "mri_1"], nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["mri_0", "mri_1", "gli_0", "gli_1"])
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys=["mri_0", "mri_1"], nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["mri_0", "mri_1", "gli_0", "gli_1"])
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.NormalizeIntensityd(keys=["mri_0", "mri_1"], nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["mri_0", "mri_1", "gli_0", "gli_1"])
            ]
        )

        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.load_image(self.data_list[item])
        if self.phase == 'train':
            data = self.train_transform(data)
        elif self.phase == 'val':
            data = self.val_transform(data)
        elif self.phase == 'test':
            data = self.test_transform(data)

        return data

    def load_image(self, file_dic):
        mri_0 = np.expand_dims(np.load(file_dic['image'][0]), 0)
        mri_1 = np.expand_dims(np.load(file_dic['image'][1]), 0)
        
        gli_0 = np.load(file_dic['label'][0])
        gli_0 = trans_brats_label(gli_0)
        
        gli_1 = np.load(file_dic['label'][1])
        gli_1 = trans_brats_label(gli_1)
        
        prompt_0 = str(file_dic['prompt'][0])
        prompt_1 = str(file_dic['prompt'][1])
        
        token = clip.tokenize(str(file_dic['prompt'][1]))
        
        return {
            'mri_0': mri_0,
            'mri_1': mri_1,
            'gli_0': gli_0,
            'gli_1': gli_1,
            'prompt_0': prompt_0,
            'prompt_1': prompt_1,
            'token': token,
            'path': os.path.split(file_dic['image'][0])[-1]
        }
        

def get_loader(
               datalist_json,
               batch_size,
               num_works,
               args=None,
               phase=None):

    files = datafold_read(datalist=datalist_json)

    # seed(12)
    # sample_num = np.ceil(len(train_files) * args.frac).astype(np.int_)
    # train_files = sample(train_files, sample_num)
    
    datasets = BraTsDataset(data_list=files, phase=phase, args=args)

    dataloader = DataLoader(datasets,
                                batch_size=batch_size,
                                num_workers=num_works,
                                pin_memory=True,
                                shuffle=True,
                                drop_last=True)

    return dataloader
    


    