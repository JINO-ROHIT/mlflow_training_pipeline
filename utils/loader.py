import torch
from utils.image_utils import read_tiff, rle_decode, rle_encode
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

from config import CFG

class HuBMAP_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labeled=True, transforms=None):
        self.df = df
        self.labeled = labeled
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.loc[index, 'image_path']
        img_height = self.df.loc[index, 'img_height']
        img_width = self.df.loc[index, 'img_width']
        img = read_tiff(img_path)
        
        if self.labeled:
            rle_mask = self.df.loc[index, 'rle']
            mask = rle_decode(rle_mask, (img_height, img_width))
            
            if self.transforms:
                data = self.transforms(image=img, mask=mask)
                img  = data['image']
                mask  = data['mask']
            
            mask = np.expand_dims(mask, axis=0)
            img = np.transpose(img, (2, 0, 1))
            
            return torch.tensor(img), torch.tensor(mask)
        
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
                
            img = np.transpose(img, (2, 0, 1))
            
            return torch.tensor(img)

data_transforms = {
    "train": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
    ]),
    
    "valid": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
}

def prepare_loaders(df, fold):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    train_dataset = HuBMAP_Dataset(train_df, transforms=data_transforms['train'])
    valid_dataset = HuBMAP_Dataset(valid_df, transforms=data_transforms['valid'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size,
                              shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=CFG.batch_size,
                              shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader