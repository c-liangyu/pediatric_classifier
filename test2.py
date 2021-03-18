import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from data.dataset import ImageDataset, ImageDataset_full
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import json, os
from model.chexpert import CheXpert_model
from metrics import AUC_ROC, F1, ACC, AUC, Precision, Recall, Specificity
from torch.optim import Adam
from torch.nn import BCELoss, BCEWithLogitsLoss
import warnings
import pandas as pd
import numpy as np
from data.dataset2 import create_loader

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

cfg_path = './config/example.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

loss_func = BCEWithLogitsLoss()

# data_dir = '/home/tungthanhlee/bdi_xray/data/images'
data_dir = 'annotations_cls_1503/images'

train_loader = create_loader(cfg.train_csv, data_dir, cfg, mode='train', dicom=False)
val_loader = create_loader(cfg.dev_csv, data_dir, cfg, mode='val', dicom=False)
test_loader = create_loader(cfg.test_csv, data_dir, cfg, mode='test', dicom=False)

metrics_dict = {'acc': ACC(), 'auc':AUC_ROC(), 'precision':Precision(), 'recall':Recall(), 'specificity':Specificity(), 'f1':F1()}

# loader_dict = {'train': train_loader, 'val': val_loader}

# model_names=['resnest', 'efficient', 'dense', 'resnest']
# ids = ['50', 'b4', '121', '101']
# ckp_paths = [
#     'experiment/train_log/ResNeSt50/21h42_130121 (end)/best1.ckpt',
#     'experiment/train_log/EfficientNet/23h_130121 (end)/best1.ckpt',
#     'experiment/train_log/DenseNet121/21h40_140121 (end)/best1.ckpt',
#     'experiment/train_log/ResNeSt101/1h_160121 (end)/best1.ckpt'
#     ]

# ckp_paths = ['experiment/train_log/ResNeSt_parallel/epoch2_iter5400.ckpt',
#              'experiment/train_log/EfficientNet_parallel/20h52_140121 (end)/epoch3_iter3200.ckpt',
#              'experiment/train_log/DenseNet121-parallel/epoch1_iter7000.ckpt',
#              'experiment/train_log/ResNeSt101-parallel/epoch2_iter7600.ckpt'
#             ]

# ckp_paths = [
#     'experiment/train_log/ResNeSt50/21h_210121_14class/epoch2_iter200.ckpt',
#     'experiment/train_log/EfficientNet/23h_210121_14classes/epoch1_iter1600.ckpt',
#     'experiment/train_log/DenseNet121/23h_230121_14classes/epoch3_iter800.ckpt',
#     'experiment/train_log/ResNeSt101/23h_210121_14class/epoch3_iter400.ckpt'
#     ]

id_leaf = [2,4,5,6,7,8]
id_obs = [0,1,2,3,4,5,6,7,8,9,10,11,12]

chexpert_model = CheXpert_model(cfg, loss_func, metrics_dict)
chexpert_model.load_ckp(cfg.ckp_path)
metrics = chexpert_model.test(val_loader)
print(cfg.backbone+'-'+cfg.id+':')
for key in metrics_dict.keys():
    if key != 'loss':
        print(key, metrics[key], metrics[key].mean())
        metrics[key] = np.append(metrics[key],metrics[key].mean())
        metrics[key] = map(lambda a: round(a, 3), metrics[key])
metrics.pop('loss')
df = pd.DataFrame.from_dict(metrics)
df.to_csv('val_result.csv', index=False,) 

metrics = chexpert_model.test(test_loader)
# if cfg.full_classes and not cfg.conditional_training:
print(cfg.backbone+'-'+cfg.id+':')
for key in metrics_dict.keys():
    if key != 'loss':
        print(key, metrics[key], metrics[key].mean())
        metrics[key] = np.append(metrics[key],metrics[key].mean())
        metrics[key] = map(lambda a: round(a, 3), metrics[key])
metrics.pop('loss')
df = pd.DataFrame.from_dict(metrics)
df.to_csv('test_result.csv', index=False,)  
# else:
#     print(cfg.backbone+'-'+cfg.id+':\n', metrics['auc'], metrics['auc'].mean(), '\n', metrics['acc'], metrics['acc'].mean())