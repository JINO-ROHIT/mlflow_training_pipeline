import segmentation_models_pytorch as smp
import torch
import transformers
import torch.optim as optim
from torch.optim import lr_scheduler

from config import CFG

JaccardLoss = smp.losses.JaccardLoss(mode='binary')
DiceLoss    = smp.losses.DiceLoss(mode='binary')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='binary', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

losses = {
    "Dice": DiceLoss,
    "Jaccard": JaccardLoss,
    "BCE": BCELoss,
    "Lovasz": LovaszLoss,
    "Tversky": TverskyLoss,
}

def get_scheduler(df, optimizer):
    
    if len(df[df['fold'] == CFG.folds_to_run[0]]) % CFG.batch_size != 0:
        num_steps = len(df[df['fold'] != CFG.folds_to_run[0]]) // CFG.batch_size + 1
    
    else:
        len(df[df['fold'] != CFG.folds_to_run[0]]) // CFG.batch_size
    
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, CFG.warmup_epochs * num_steps, CFG.epochs * num_steps)
        
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, threshold=0.0001, min_lr=1e-6)
    elif CFG.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        
    return scheduler

def get_optimizer(model, optimizer_name=CFG.optimizer):
    if CFG.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    
    elif CFG.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        
    return optimizer