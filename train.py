import sys
sys.path.extend([
    "/usr/lib/python3/dist-packages"
])

import numpy as np
import pandas as pd
import time
import os
import random
import torch
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import copy
import gc
from loguru import logger
import mlflow
from argparse import ArgumentParser

from config import CFG
from utils.modelling import train_one_epoch, valid_one_epoch, build_model
from utils.loader import prepare_loaders
from utils.hyper_params import get_optimizer, get_scheduler

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.success('> SEEDING DONE')


def main():
    set_seed(CFG.seed)
    parser = ArgumentParser(description="HUBMAP segmentation")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="fold number",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="experiment id",
    )
    parser.add_argument(
        "--path",
        type=str,
        default = CFG.base_path,
        help="data path",
    )

    args = parser.parse_args()
    fold, path = args.fold, args.path

    mlflow_experiment_id = args.id

    df = pd.read_csv(os.path.join(path, "train.csv"))
    kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = fold
    df['image_path'] = df['id'].apply(lambda x: os.path.join(CFG.base_path, 'train_images', str(x) + '.tiff'))

    train_loader, valid_loader = prepare_loaders(df, fold = fold)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    model = build_model()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(df, optimizer)

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1

    mlflow.log_params({key: value for key, value in vars(CFG).items() if not key.startswith('__') and not callable(key)})
    for epoch in range(1, CFG.epochs + 1): 
        gc.collect()
        logger.info(f'Epoch {epoch}/{CFG.epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader = train_loader, 
                                           device=CFG.device, epoch=epoch)
        
        val_loss, val_scores = valid_one_epoch(model, optimizer, valid_loader, 
                                                 device=CFG.device, 
                                                 epoch=epoch)
        val_dice, val_jaccard = val_scores

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("validation_loss", val_loss)
        mlflow.log_metric("dice", val_dice)
        mlflow.log_metric("jaccard", val_jaccard)
        
        logger.info(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        
        # deep copy the model
        if val_dice >= best_dice:
            logger.success(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            logger.info(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            logger.info(f"Model Saved")
            
        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)
    
    end = time.time()
    time_elapsed = end - start
    logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logger.success("Best Score: {:.4f}".format(best_dice))

    mlflow.sklearn.log_model(model, PATH)

if __name__ == "__main__":
    main()