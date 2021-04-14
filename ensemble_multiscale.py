import os

from efficientnet_pytorch import model
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
from data.dataset import create_loader
import torch
import argparse
from model.utils import tensor2numpy

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

cfg_path = './config/ensemble_config.json'
# cfg_path = './config/example2.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

if isinstance(cfg.batch_size, list) and isinstance(cfg.long_side, list):
    list_batch = cfg.batch_size
    list_res = cfg.long_side
elif isinstance(cfg.batch_size, int) and isinstance(cfg.long_side, int):
    list_batch = [cfg.batch_size]
    list_res = [cfg.long_side]
else:
    raise Exception("'batch_size' and 'long_side' in config file should be same instance!!!")

loss_func = BCEWithLogitsLoss()

# data_dir = '/home/tungthanhlee/bdi_xray/data/images'
data_dir = '/home/dual1/thanhtt/assigned_jpeg'
metrics_dict = {'acc': ACC(), 'auc':AUC(), 'precision':Precision(), 'recall':Recall(), 'specificity':Specificity(), 'f1':F1()}

model_names=[
    'dense',
    'dense',
    'dense',
    # 'resnet',
    # 'dense',
    # 'efficient',
    #'resnest'
    ]
ids = [
    '121',
    '121',
    '121',
    # '101',
    # '169',
    # 'b4',
    #'101'
    ]
ckp_paths = [
    'experiment/DenseNet121_data2203_256/checkpoint/best.ckpt',
    'experiment/DenseNet121_parallel_data2203/checkpoint/best.ckpt',
    'experiment/DenseNet121_data2203_1024/checkpoint/best.ckpt'
    # 'experiment/Resnet101_parallel_data2203/checkpoint/best.ckpt',
    # 'experiment/Dense169_parallel_data2203/checkpoint/best.ckpt',
    # 'experiment/EfficientB4_parallel_data2203/checkpoint/best.ckpt',
    # 'experiment/ResneSt101_parallel_data2203/checkpoint/best.ckpt'
    ]

id_leaf = [2,4,5,6,7,8]
id_obs = [0,1,2,3,4,5,6,7,8,9,10,11,12]

preds_stack_val = []
labels_stack_val = []
preds_stack_test = []
labels_stack_test = []

for i in range(len(list_batch)):
    cfg.batch_size = list_batch[i]
    cfg.long_side = list_res[i]
    train_loader = create_loader(cfg.train_csv, data_dir, cfg, mode='train', dicom=False, type=cfg.type)
    val_loader = create_loader(cfg.dev_csv, data_dir, cfg, mode='val', dicom=False, type=cfg.type)
    test_loader = create_loader(cfg.test_csv, data_dir, cfg, mode='test', dicom=False, type=cfg.type)

    print(f'{model_names[i]}-{ids[i]}:' )
    cfg.backbone = model_names[i]
    cfg.id = ids[i]
    cfg.ckp_path = ckp_paths[i]
    chexpert_model = CheXpert_model(cfg, loss_func, metrics_dict)
    chexpert_model.load_ckp(cfg.ckp_path)
    preds, labels, _ = chexpert_model.predict_loader(
        val_loader, ensemble=False, cal_loss=False)
    preds_stack_val.append(preds)
    labels_stack_val.append(labels)
    preds, labels, _ = chexpert_model.predict_loader(
        test_loader, ensemble=False, cal_loss=False)
    preds_stack_test.append(preds)
    labels_stack_test.append(labels)

preds_val = torch.mean(torch.stack(preds_stack_val, dim=0), dim=0)
labels_val = torch.mean(torch.stack(labels_stack_val, dim=0), dim=0)
preds_test = torch.mean(torch.stack(preds_stack_test, dim=0), dim=0)
labels_test = torch.mean(torch.stack(labels_stack_test, dim=0), dim=0)
auc_opt = AUC_ROC()
thresh_val = auc_opt(preds_val, labels_val, thresholding=True)
thresh_val = torch.Tensor(thresh_val).float().cuda()
running_metrics_val = dict.fromkeys(metrics_dict.keys(), 0.0)
running_metrics_test = dict.fromkeys(metrics_dict.keys(), 0.0)
for key in list(metrics_dict.keys()):
    if key in ['f1', 'precision', 'recall', 'specificity', 'acc']:
        running_metrics_val[key] = tensor2numpy(metrics_dict[key](
            preds_val, labels_val, thresh_val))
        running_metrics_test[key] = tensor2numpy(metrics_dict[key](
            preds, labels_test, thresh_val))
    else:
        running_metrics_val[key] = tensor2numpy(metrics_dict[key](
            preds_val, labels_val))
        running_metrics_test[key] = tensor2numpy(metrics_dict[key](
            preds_test, labels_test))

for key in metrics_dict.keys():
    if key != 'loss':
        if key == 'auc' and cfg.type == 'chexmic':
            print(key, running_metrics_val[key], running_metrics_val[key][id_obs].mean())
            running_metrics_val[key] = np.append(running_metrics_val[key],running_metrics_val[key][id_obs].mean())
        else:
            print(key, running_metrics_val[key], running_metrics_val[key].mean())
            running_metrics_val[key] = np.append(running_metrics_val[key],running_metrics_val[key].mean())
        running_metrics_val[key] = map(lambda a: round(a, 3), running_metrics_val[key])
running_metrics_val.pop('loss')
df = pd.DataFrame.from_dict(running_metrics_val)
df.to_csv(f'val'+'_result.csv', index=False,) 

for key in metrics_dict.keys():
    if key != 'loss':
        print(key, running_metrics_test[key], running_metrics_test[key].mean())
        running_metrics_test[key] = np.append(running_metrics_test[key],running_metrics_test[key].mean())
        running_metrics_test[key] = map(lambda a: round(a, 3), running_metrics_test[key])
running_metrics_test.pop('loss')
df = pd.DataFrame.from_dict(running_metrics_test)
df.to_csv('test'+'_result.csv', index=False,)