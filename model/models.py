import torch.nn as nn
import torch
import torch.nn.functional as F
import os

def save_dense_backbone(model, ckp_path):
    if os.path.exists(os.path.dirname(ckp_path)):
        torch.save({'state_dict': model.features.state_dict(),
                    'epoch': 0, 'iter': 0}, ckp_path)
    else:
        print("Save path not exist!!!")

def load_dense_backbone(model, ckp_path, device, strict):
    if os.path.exists(os.path.dirname(ckp_path)):
        ckp = torch.load(ckp_path, map_location=device)
        model.features.load_state_dict(ckp['state_dict'], strict=strict)
    else:
        print("Save path not exist!!!")

def save_resnet_backbone(model, ckp_path):
    if os.path.exists(os.path.dirname(ckp_path)):
        save_model = model
        delattr(save_model,'avgpool')
        delattr(save_model,'fc')
        torch.save({'state_dict': save_model.state_dict(),
                    'epoch': 0, 'iter': 0}, ckp_path)
    else:
        print("Save path not exist!!!")

def load_resnet_backbone(model, ckp_path, device, strict):
    if os.path.exists(os.path.dirname(ckp_path)):
        ckp = torch.load(ckp_path, map_location=device)
        model.load_state_dict(ckp['state_dict'], strict=strict)
    else:
        print("Save path not exist!!!")

def save_efficient_backbone(model, ckp_path):
    if os.path.exists(os.path.dirname(ckp_path)):
        torch.save({'state_dict': model.features.state_dict(),
                    'epoch': 0, 'iter': 0}, ckp_path)
    else:
        print("Save path not exist!!!")

def load_efficient_backbone(model, ckp_path, device, strict):

    if os.path.exists(os.path.dirname(ckp_path)):
        ckp = torch.load(ckp_path, map_location=device)
        model.features.load_state_dict(ckp['state_dict'], strict=strict)
    else:
        print("Save path not exist!!!")

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x):
        y = []
        for module in self:
            y.append(nn.Sigmoid()(module(x)))
        # y = torch.stack(y).max(0)[0]  # max ensemble
        y = torch.stack(y, 0).mean(0)  # mean ensemble
        # y = torch.cat(y, 1)  # nms ensemble

        return y

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ResNeSt_parallel(nn.Module):
    def __init__(self, pre_model, num_classes):
        """ResNeSt-based model - not split the head of the model, instead use one linear layer to return directly outputs
        Args:
            pre_model (torch.nn.module): predenfined ResNeSt backbone model.
            num_classes (int): number of classes.
        """
        super(ResNeSt_parallel, self).__init__()
        self.conv1 = pre_model.conv1
        self.bn1 = pre_model.bn1
        self.relu = pre_model.relu
        self.maxpool = pre_model.maxpool
        self.layer1 = pre_model.layer1
        self.layer2 = pre_model.layer2
        self.layer3 = pre_model.layer3
        self.layer4 = pre_model.layer4
        self.avgpool = pre_model.avgpool
        self.num_features = pre_model.fc.in_features
        self.fc = nn.Linear(in_features=self.num_features,
                            out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class Dense_parallel(nn.Module):
    def __init__(self, pre_model, num_classes):
        """DenseNet-based model - not split the head of the model, instead use one linear layer to return directly outputs
        Args:
            pre_model (torch.nn.module): predenfined DenseNet backbone model.
            num_classes (int): number of classes.
        """
        super(Dense_parallel, self).__init__()
        self.features = pre_model.features
        self.num_features = pre_model.classifier.in_features
        self.fc = nn.Linear(in_features=self.num_features,
                            out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class Efficient_parallel(nn.Module):
    def __init__(self, pre_model, num_classes):
        """EfficientNet-based model - not split the head of the model, instead use one linear layer to return directly outputs
        Args:
            pre_model (torch.nn.module): predenfined EfficientNet backbone model.
            num_classes (int): number of classes.
        """
        super(Efficient_parallel, self).__init__()
        self._conv_stem = pre_model._conv_stem
        self._bn0 = pre_model._bn0
        self._blocks = pre_model._blocks
        self._conv_head = pre_model._conv_head
        self._bn1 = pre_model._bn1
        self._avg_pooling = pre_model._avg_pooling
        self._dropout = pre_model._dropout
        self.num_features = pre_model._fc.in_features
        self._fc = nn.Linear(in_features=self.num_features,
                             out_features=num_classes, bias=True)

    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        for block in self._blocks:
            x = block(x)
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._avg_pooling(x)
        x = self._dropout(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = self._fc(x)
        return x