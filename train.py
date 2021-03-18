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

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")


cfg_path = './config/example.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

# if cfg.full_classes:
#     data_class = ImageDataset_full
# else:
#     data_class = ImageDataset

data_class = ImageDataset_full

train_loader = DataLoader(data_class(cfg.train_csv, cfg, mode='train'),
                          num_workers=4,drop_last=True,shuffle=True,
                          batch_size=cfg.batch_size)
val_loader = DataLoader(data_class(cfg.dev_csv, cfg, mode='dev'),
                        num_workers=4,drop_last=False,shuffle=False,
                        batch_size=cfg.batch_size)

# loss_func = BCELoss()
loss_func = BCEWithLogitsLoss()

# class_weights = [3.]*len(cfg.num_classes)
# beta = 0.99
# class_weights = []
# for num_img in train_loader.dataset.class_dist:
    # class_weights.append((1-beta)/(1-beta**num_img))
    # class_weights.append(train_loader.dataset._num_image/num_img)
# class_weights = np.array(class_weights) / np.sum(class_weights) * len(cfg.num_classes)
# print(class_weights)
# if cfg.device == 'cpu':
#     device = torch.device("cpu")
# else:
#     device = torch.device("cuda:"+str(cfg.device))
# loss_func = BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights, device=device))

metrics_dict = {'acc': ACC(), 'auc':AUC(), 'f1':F1(), 'precision':Precision(), 'recall':Recall()}
loader_dict = {'train': train_loader, 'val': val_loader}

chexpert_model = CheXpert_model(cfg, loss_func, metrics_dict)

# chexpert_model.load_ckp(cfg.ckp_path)
# chexpert_model.freeze_backbone()

writer = SummaryWriter(os.path.join('experiment', cfg.log_dir))
ckp_dir = os.path.join('experiment', cfg.log_dir, 'checkpoint')

chexpert_model.train(train_loader, val_loader, epochs=cfg.epochs, iter_log=cfg.iter_log, writer=writer, eval_metric='auc', ckp_dir=ckp_dir)