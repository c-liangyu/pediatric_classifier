from easydict import EasyDict as edict
import json, os, torch
from metrics import F1, ACC, AUC, Precision, Recall, Specificity
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from model.chexpert import CheXpert_model
from data.dataset import create_loader
import warnings

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

cfg_path = './config/finetune_config.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

# data_dir = '/home/tungthanhlee/bdi_xray/data/images'
data_dir = '/home/tungthanhlee/thanhtt/assigned_jpeg'

train_loader = create_loader(cfg.train_csv, data_dir, cfg, mode='train', dicom=False, type=cfg.type)
val_loader = create_loader(cfg.dev_csv, data_dir, cfg, mode='val', dicom=False, type=cfg.type)

# loss_func = BCELoss()
loss_func = BCEWithLogitsLoss()

metrics_dict = {'auc':AUC(), 'sensitivity':Recall(), 'specificity':Specificity(), 'f1':F1()}
loader_dict = {'train': train_loader, 'val': val_loader}

chexpert_model = CheXpert_model(cfg, loss_func, metrics_dict)

chexpert_model.load_backbone(cfg.ckp_path, strict=False)
# chexpert_model.freeze_backbone()

writer = SummaryWriter(os.path.join('experiment', cfg.log_dir))
ckp_dir = os.path.join('experiment', cfg.log_dir, 'checkpoint')

lr_hist = chexpert_model.train(train_loader, val_loader, epochs=cfg.epochs, iter_log=cfg.iter_log, eval_metric='auc', ckp_dir=ckp_dir, use_lr_sch=False)


# plt.plot(range(cfg.epochs),lr_hist)
# plt.show()