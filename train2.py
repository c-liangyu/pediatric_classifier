from torch.nn.modules import loss
from data.dataset import ImageDataset, ImageDataset_full
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import json, os, torch
from metrics import F1, ACC, AUC, Precision, Recall
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from model.chexpert import CheXpert_model
from data.dataset2 import create_loader
import warnings
import numpy as np
import argparse

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()


cfg_path = './config/example.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

if cfg.distributed:
    cfg.device = args.local_rank
    torch.cuda.set_device(cfg.device)
    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    cfg.world_size = torch.distributed.get_world_size()

# data_dir = '/home/tungthanhlee/bdi_xray/data/images'
data_dir = 'annotations_cls_1803/images'

train_loader = create_loader(cfg.train_csv, data_dir, cfg, mode='train', dicom=False, type=cfg.type)
val_loader = create_loader(cfg.dev_csv, data_dir, cfg, mode='val', dicom=False, type=cfg.type)

# loss_func = BCELoss()
loss_func = BCEWithLogitsLoss()

metrics_dict = {'acc': ACC(), 'auc':AUC(), 'f1':F1(), 'precision':Precision(), 'recall':Recall()}
loader_dict = {'train': train_loader, 'val': val_loader}

chexpert_model = CheXpert_model(cfg, loss_func, metrics_dict)

# chexpert_model.load_ckp(cfg.ckp_path)
# chexpert_model.freeze_backbone()

writer = SummaryWriter(os.path.join('experiment', cfg.log_dir))
ckp_dir = os.path.join('experiment', cfg.log_dir, 'checkpoint')

chexpert_model.train(train_loader, val_loader, epochs=cfg.epochs, iter_log=cfg.iter_log, writer=writer, eval_metric='auc', ckp_dir=ckp_dir)