import torch
import multiprocessing as mp

class CFG:
    seed = 0
    batch_size = 4
    head = "UNet"
    backbone = "efficientnet-b0"
    img_size = [256, 256]
    lr = 1e-3
    scheduler = 'CosineAnnealingLR'
    epochs = 2
    warmup_epochs = 2
    n_folds = 5
    folds_to_run = [0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_path = "hubmap-organ-segmentation"
    num_workers = mp.cpu_count()
    num_classes = 1
    n_accumulate = max(1, 16//batch_size)
    loss = 'Dice'
    optimizer = 'Adam'
    weight_decay = 1e-6