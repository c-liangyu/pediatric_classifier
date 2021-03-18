import torch.nn as nn
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from model.models import ResNeSt_parallel, Efficient_parallel, Dense_parallel
# from resnest.torch import resnest50, resnest101, resnest200, resnest269
# from efficientnet_pytorch import EfficientNet
from torchvision.models import densenet121, densenet161, densenet169, densenet201, resnet18, resnet34, resnet50, resnet101


def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))

def get_model(cfg):
    if cfg.backbone == 'resnest':
        childs_cut = 9
        if cfg.id == '50':
            pre_name = resnest50
        elif cfg.id == '101':
            pre_name = resnest101
        elif cfg.id == '200':
            pre_name = resnest200
        else:
            pre_name = resnest269
        pre_model = pre_name(pretrained=cfg.pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        model = ResNeSt_parallel(pre_model, len(cfg.num_classes))
    elif cfg.backbone == 'efficient' or cfg.backbone == 'efficientnet':
        childs_cut = 6
        pre_name = 'efficientnet-'+cfg.id
        if cfg.pretrained:
            pre_model = EfficientNet.from_pretrained(pre_name)
        else:
            pre_model = EfficientNet.from_name(pre_name)
        for param in pre_model.parameters():
            param.requires_grad = True
        model = Efficient_parallel(pre_model, len(cfg.num_classes))
    elif cfg.backbone == 'dense' or cfg.backbone == 'densenet':
        childs_cut = 2
        if cfg.id == '121':
            pre_name = densenet121
        elif cfg.id == '161':
            pre_name = densenet161
        elif cfg.id == '169':
            pre_name = densenet169
        else:
            pre_name = densenet201
        pre_model = pre_name(pretrained=cfg.pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        model = Dense_parallel(pre_model, len(cfg.num_classes))
    else:
        raise Exception("Not support this model!!!!")
    return model, childs_cut

def get_model_new(cfg):
    if cfg.backbone == 'resnest':
        childs_cut = 9
        if cfg.id == '50':
            pre_name = resnest50
        elif cfg.id == '101':
            pre_name = resnest101
        elif cfg.id == '200':
            pre_name = resnest200
        else:
            pre_name = resnest269
        model = pre_name(pretrained=cfg.pretrained)
        for param in model.parameters():
            param.requires_grad = True
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features,
                            out_features=len(cfg.num_classes), bias=True)
        
    elif cfg.backbone == 'efficient' or cfg.backbone == 'efficientnet':
        childs_cut = 6
        pre_name = 'efficientnet-'+cfg.id
        if cfg.pretrained:
            model = EfficientNet.from_pretrained(pre_name)
        else:
            model = EfficientNet.from_name(pre_name)
        for param in model.parameters():
            param.requires_grad = True        
        num_features = model._fc.in_features
        model._fc = nn.Linear(in_features=num_features,
                             out_features=len(cfg.num_classes), bias=True)

    elif cfg.backbone == 'dense' or cfg.backbone == 'densenet':
        childs_cut = 2
        if cfg.id == '121':
            pre_name = densenet121
        elif cfg.id == '161':
            pre_name = densenet161
        elif cfg.id == '169':
            pre_name = densenet169
        else:
            pre_name = densenet201
        model = pre_name(pretrained=cfg.pretrained)
        for param in model.parameters():
            param.requires_grad = True
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features=num_features,
                            out_features=len(cfg.num_classes), bias=True)
    elif cfg.backbone == 'resnet':
        childs_cut = 0
        if cfg.id == '34':
            pre_name = resnet34
        elif cfg.id == '18':
            pre_name = resnet18
        elif cfg.id == '50':
            pre_name = resnet50
        else:
            pre_name = resnet101
        model = pre_name(pretrained=cfg.pretrained)
        for param in model.parameters():
            param.requires_grad = True
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features,
                            out_features=len(cfg.num_classes), bias=True)
    else:
        raise Exception("Not support this model!!!!")
    return model, childs_cut

def get_str(metrics, mode, s):
    for key in list(metrics.keys()):
        if key == 'loss':
            s += "{}_{} {:.3f} - ".format(mode, key, metrics[key])
        else:
            metric_str = ' '.join(
                map(lambda x: '{:.5f}'.format(x), metrics[key]))
            s += "{}_{} {} - ".format(mode, key, metric_str)
    s = s[:-2] + '\n'
    return s

def tensor2numpy(input_tensor):
    # device cuda Tensor to host numpy
    return input_tensor.cpu().detach().numpy()
