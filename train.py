from easydict import EasyDict as edict
import json, os, torch
from metrics import F1, ACC, AUC, Precision, Recall
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from model.chexpert import CheXpert_model
from data.dataset import create_loader
import warnings

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

# cfg_path = './config/chexmic_config.json'
cfg_path = './config/example.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

data_dir = '/home/tungthanhlee/thanhtt/assigned_jpeg'

train_loader = create_loader(cfg.train_csv, data_dir, cfg, mode='train')
val_loader = create_loader(cfg.dev_csv, data_dir, cfg, mode='val')

loss_func = BCEWithLogitsLoss()

metrics_dict = {'acc': ACC(), 'auc':AUC(), 'f1':F1(), 'precision':Precision(), 'recall':Recall()}
loader_dict = {'train': train_loader, 'val': val_loader}

chexpert_model = CheXpert_model(cfg, loss_func, metrics_dict)
# print(chexpert_model.model)

# chexpert_model.load_ckp(cfg.ckp_path)
# print(chexpert_model.model.state_dict().keys())
# chexpert_model.save_backbone('experiment/Resnet101_chexmic/checkpoint/backbone.ckpt')
# chexpert_model.freeze_backbone()

writer = SummaryWriter(os.path.join('experiment', cfg.log_dir))
ckp_dir = os.path.join('experiment', cfg.log_dir, 'checkpoint')

chexpert_model.train(train_loader, val_loader, epochs=cfg.epochs, iter_log=cfg.iter_log, eval_metric='auc', ckp_dir=ckp_dir, resume=False)

if cfg.type == 'chexmic':
    chexpert_model.save_backbone(os.path.join(ckp_dir, 'backbone.ckpt'))