from easydict import EasyDict as edict
import json, os, torch
from metrics import F1, ACC, AUC, Precision, Recall
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from model.chexpert import CheXpert_model
from data.dataset import create_loader
import warnings
import argparse

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()


# cfg_path = './config/chexmic_config.json'
cfg_path = './config/example.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

if cfg.distributed:
    print('here')
    cfg.device = args.local_rank
    torch.cuda.set_device(cfg.device)
    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    cfg.world_size = torch.distributed.get_world_size()
else:
    torch.cuda.set_device(cfg.device)

data_dir = '/home/tungthanhlee/thanhtt/assigned_jpeg'

train_loader = create_loader(cfg.train_csv, data_dir, cfg, mode='train', dicom=False, type=cfg.type)
val_loader = create_loader(cfg.dev_csv, data_dir, cfg, mode='val', dicom=False, type=cfg.type)

# loss_func = BCELoss()
# loss_func = BCEWithLogitsLoss()
loss_func = MSELoss()

metrics_dict = {'acc': ACC(), 'auc':AUC(), 'f1':F1(), 'precision':Precision(), 'recall':Recall()}
loader_dict = {'train': train_loader, 'val': val_loader}

#------------------------------- additional config for ensemble ---------------------------------------
model_names=[
    'dense',
    'resnet',
    'dense',
    # 'efficient',
    #'resnest'
    ]
ids = [
    '121',
    '101',
    '169',
    # 'b4',
    #'101'
    ]

# ckp_paths = [
#     'experiment/DenseNet121_data2203_finetune_chexpmic_cutmix/checkpoint/best.ckpt',
#     'experiment/Resnet101_data2203_finetune_chexpmic_cutmix/checkpoint/best.ckpt',
#     'experiment/DenseNet169_data2203_finetune_chexpmic_cutmix/checkpoint/best.ckpt',
# ]

# ckp_paths = [
#     'experiment/DenseNet121_data2203_finetune_chexpmic_mixup/checkpoint/best.ckpt',
#     'experiment/Resnet101_data2203_finetune_chexpmic_mixup/checkpoint/best.ckpt',
#     'experiment/DenseNet169_data2203_finetune_chexpmic_mixup/checkpoint/best.ckpt'
# ]

ckp_paths = [
    'experiment/DenseNet121_data2203_finetune_chexpmic/checkpoint/best.ckpt',
    'experiment/Resnet101_data2203_finetune_chexpmic/checkpoint/best.ckpt',
    'experiment/DenseNet169_data2203_finetune_chexpmic/checkpoint/best.ckpt',
    ]

cfg.backbone = model_names
cfg.id = ids
cfg.ckp_path = ckp_paths
#------------------------------------------------------------------------------------------------------

chexpert_model = CheXpert_model(cfg, loss_func, metrics_dict)
chexpert_model.stacking(train_loader, val_loader, epochs=5)
# print(chexpert_model.model)

# chexpert_model.load_ckp(cfg.ckp_path)
# print(chexpert_model.model.state_dict().keys())
# chexpert_model.save_backbone('experiment/Resnet101_chexmic/checkpoint/backbone.ckpt')
# chexpert_model.freeze_backbone()

# writer = SummaryWriter(os.path.join('experiment', cfg.log_dir))
# ckp_dir = os.path.join('experiment', cfg.log_dir, 'checkpoint')

# chexpert_model.train(train_loader, val_loader, epochs=cfg.epochs, iter_log=cfg.iter_log, eval_metric='auc', ckp_dir=ckp_dir, resume=False)

# if cfg.type == 'chexmic':
#     chexpert_model.save_backbone(os.path.join(ckp_dir, 'backbone.ckpt'))