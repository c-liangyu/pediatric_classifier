import pandas as pd
import torch
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Resize, HorizontalFlip, RandomBrightnessContrast, OneOf, Blur, MotionBlur, IAAAdditiveGaussianNoise, ShiftScaleRotate
from albumentations.pytorch import ToTensor
from albumentations import Lambda, Rotate
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data.utils import transform
import time

def preprocess(img_size): return Compose(
    [Resize(img_size[0], img_size[1]), ToTensor()])

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def read_xray(path):

    dicom = pydicom.read_file(path)

    # VOI LUT is used to transform raw DICOM data to "human-friendly" view
    data = apply_voi_lut(dicom.pixel_array, dicom)

    # depending on this value, X-ray may look inverted - fix that:
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


def correct_path(path): return path if os.path.isfile(
    path) else path[:-3]+'dicom'


class Pediatric_dicom(Dataset):
    def __init__(self, label_path, data_dir, cfg, mode='train', transforms=None, type=None, dicom=False):
        """Image generator for conditional training and finetuning parent samples
        Args:
            label_path (str): path to .csv file contains img paths and class labels
            cfg (str): configuration file.
            mode (str, optional): define which mode you are using. Defaults to 'train'.
        """
        self.data_dir = data_dir
        self.type = type
        self.cfg = cfg
        self.dicom = dicom
        self.df = pd.read_csv(label_path)
        if self.type == 'pediatric':
            self.disease_classes = ["Other opacity", "Reticulonodular opacity", "Peribronchovascular interstitial opacity", "Diffuse aveolar opacity", "Lung hyperinflation",
                                    "Consolidation", "Bronchial thickening", "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Pneumonia"]
            self.id_leaf = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
            if self.cfg.conditional_training:
                self.df = self.df[self.df[self.disease_classes[7]] != 1.0]
        elif self.type == 'sub_pediatric':
            self.disease_classes = ["Other opacity", "Reticulonodular opacity", "Peribronchovascular interstitial opacity",
                                    "Bronchial thickening", "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Pneumonia"]
            self.id_leaf = [0, 1, 2, 3, 5, 6, 7, 8, 9]
            if self.cfg.conditional_training:
                self.df = self.df[self.df[self.disease_classes[4]] != 1.0]
        elif self.type == 'chexmic':
            self.disease_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                                    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
            self.id_leaf = [2, 4, 5, 6, 7, 8]
            if self.cfg.conditional_training:
                self.id_parent = [0, 1, 3, 9, 10, 11, 12, 13]
                for id_par in self.id_parent:
                    self.df = self.df[self.df[self.disease_classes[id_par]] == 1.0]
        else:
            self.disease_classes = ["No finding"]
        if self.type == 'chexmic':
            self.img_ids = self.df.Path.unique()
        else:
            self.img_ids = self.df.image_id.unique()
        self.transforms = transforms
        self.mode = mode
        # self.img_size = (self.cfg.long_side, self.cfg.long_side)
        self.n_data = len(self.img_ids)
        self._num_image = self.n_data
        print(f"total images in {mode} set: {self.n_data}")

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        if self.dicom:
            img_path = os.path.join(
                self.data_dir, img_id+'.dcm')
            img_path = correct_path(img_path)
            image = read_xray(img_path)
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            if self.type == 'chexmic':
                img_path = os.path.join(self.data_dir, img_id)
            else:
                img_path = os.path.join(
                    self.data_dir, img_id+'.jpg')
            # print(img_path)
            image = cv2.imread(img_path, 0)
        if self.type == 'pediatric' or self.type == 'sub_pediatric':
            label = self.df.iloc[idx].values
            label = label[1:]
            label = np.array(label).astype(np.float32)
        elif self.type == 'chexmic':
            self.df = self.df.fillna(0)
            label = self.df.iloc[idx].values
            if self.mode == 'train':
                label = label[1:]
            else:
                label = label[5:]
            label = [random.uniform(self.smooth_range[0], self.smooth_range[1])
                     if x == -1.0 else x for x in label]
            label = np.array(label).astype(np.float32)
        else:
            label = torch.Tensor([self.df['No finding'][idx]])
        # st = time.perf_counter()
        if self.mode != 'train' and self.cfg.tta:
            image_list = self.transforms(image)
            image_list = [torch.Tensor(transform(image, self.cfg)).float() for image in image_list]
            image = torch.stack(image_list, dim=0)
        else:
            if self.mode == 'train':
                image = self.transforms(image=image)['image']
            image = transform(image, self.cfg)
            image = torch.Tensor(image).float()
        # end = time.perf_counter()
        # print(end-st)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # img_size = (self.cfg.long_side, self.cfg.long_side)
        # image = preprocess(img_size)(image=image)['image']
        # print(type(image), type(label))
        # print(image.shape, label.shape)
        if self.mode == 'train':
            return image, label
        else:
            return image, label, img_id

class Pediatric_dicom_cutmix(Dataset):
    def __init__(self, label_path, data_dir, cfg, mode='train', transforms=None, type=None, dicom=False):
        """Image generator for conditional training and finetuning parent samples
        Args:
            label_path (str): path to .csv file contains img paths and class labels
            cfg (str): configuration file.
            mode (str, optional): define which mode you are using. Defaults to 'train'.
        """
        self.data_dir = data_dir
        self.type = type
        self.cfg = cfg
        self.dicom = dicom
        self.df = pd.read_csv(label_path)
        if self.type == 'pediatric':
            self.disease_classes = ["Other opacity", "Reticulonodular opacity", "Peribronchovascular interstitial opacity", "Diffuse aveolar opacity", "Lung hyperinflation",
                                    "Consolidation", "Bronchial thickening", "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Pneumonia"]
            self.id_leaf = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
            if self.cfg.conditional_training:
                self.df = self.df[self.df[self.disease_classes[7]] != 1.0]
        elif self.type == 'sub_pediatric':
            self.disease_classes = ["Other opacity", "Reticulonodular opacity", "Peribronchovascular interstitial opacity",
                                    "Bronchial thickening", "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Pneumonia"]
            self.id_leaf = [0, 1, 2, 3, 5, 6, 7, 8, 9]
            if self.cfg.conditional_training:
                self.df = self.df[self.df[self.disease_classes[4]] != 1.0]
        elif self.type == 'chexmic':
            self.disease_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                                    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
            self.id_leaf = [2, 4, 5, 6, 7, 8]
            if self.cfg.conditional_training:
                self.id_parent = [0, 1, 3, 9, 10, 11, 12, 13]
                for id_par in self.id_parent:
                    self.df = self.df[self.df[self.disease_classes[id_par]] == 1.0]
        else:
            self.disease_classes = ["No finding"]
        if self.type == 'chexmic':
            self.img_ids = self.df.Path.unique()
        else:
            self.img_ids = self.df.image_id.unique()
        self.transforms = transforms
        self.mode = mode
        # self.img_size = (self.cfg.long_side, self.cfg.long_side)
        self.n_data = len(self.img_ids)
        self._num_image = self.n_data
        print(f"total images in {mode} set: {self.n_data}")

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        # image, label = self.get_aug_image(index)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # img_size = (self.cfg.long_side, self.cfg.long_side)
        # image = preprocess(img_size)(image=image)['image']
        if self.mode == 'train':
            # image, label = self.load_cutmix_image_and_label(idx, num_mix=4)
            image, label = self.load_mixup_image_and_label(idx)
        else:
            image, label = self.get_aug_image(idx)
        
        return image, label

    def get_aug_image(self, idx):

        image, label = self.load_image_and_label(idx)
        if self.mode != 'train' and self.cfg.tta:
            image_list = self.transforms(image)
            image_list = [torch.Tensor(transform(image, self.cfg)).float() for image in image_list]
            image = torch.stack(image_list, dim=0)
        else:
            if self.mode == 'train':
                image = self.transforms(image=image)['image']
            image = transform(image, self.cfg)
            image = torch.Tensor(image).float()
        
        return image, label

    def load_image_and_label(self, idx):

        img_id = self.img_ids[idx]
        if self.dicom:
            img_path = os.path.join(
                self.data_dir, img_id+'.dcm')
            img_path = correct_path(img_path)
            image = read_xray(img_path)
        else:
            if self.type == 'chexmic':
                img_path = os.path.join(self.data_dir, img_id)
            else:
                img_path = os.path.join(
                    self.data_dir, img_id+'.jpg')
            image = cv2.imread(img_path, 0)
        if self.type == 'pediatric' or self.type == 'sub_pediatric':
            label = self.df.iloc[idx].values
            label = label[1:]
            label = np.array(label).astype(np.float32)
        elif self.type == 'chexmic':
            self.df = self.df.fillna(0)
            label = self.df.iloc[idx].values
            if self.mode == 'train':
                label = label[1:]
            else:
                label = label[5:]
            label = [random.uniform(self.smooth_range[0], self.smooth_range[1])
                     if x == -1.0 else x for x in label]
            label = np.array(label).astype(np.float32)
        else:
            label = torch.Tensor([self.df['No finding'][idx]])
        
        return image, label

    def load_mixup_image_and_label(self, index):
        if self.cfg.beta > 0:
            lam = np.random.beta(self.cfg.beta, self.cfg.beta)
        else:
            lam = 1
        image, label = self.get_aug_image(index)
        rand_index = random.choice(range(self.n_data))
        r_image, r_label = self.get_aug_image(rand_index)
        return lam*image+(1-lam)*r_image, lam*label+(1-lam)*r_label

    def load_cutmix_image_and_label(self, idx, num_mix):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        image, label = self.get_aug_image(idx)
        # print(type(image), type(label))
        for _ in range(num_mix):
            r = np.random.rand(1)
            if self.cfg.beta <= 0 or r > self.cfg.cutmix_prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.cfg.beta, self.cfg.beta)
            rand_index = random.choice(range(self.n_data))

            image_rand, label_rand = self.get_aug_image(rand_index)

            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            image[:, bbx1:bbx2, bby1:bby2] = image_rand[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            label = label * lam + label_rand * (1. - lam)

        return image, label

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def test_time_aug(image):
    img1 = image.copy()
    img2 = HorizontalFlip(p=1)(image=image)['image']
    img3 = rotate_image(img1, 5)
    img4 = rotate_image(img1, -5)
    img5 = rotate_image(img2, 5)
    img6 = rotate_image(img2, -5)
    return [img1, img2, img3, img4, img5, img6]

# def test_time_aug(image):
#     img1 = image.copy()
#     img2 = HorizontalFlip(p=1)(image=image)['image']
#     return [img1, img2]

def create_loader(label_path, data_dir, cfg, mode='train', dicom=False, type=None):

    transforms_aug = Compose([
        HorizontalFlip(),
        # RandomBrightnessContrast(
        #     always_apply=True, brightness_limit=0.2, contrast_limit=0.2),
        # OneOf([Blur(blur_limit=2, p=0.6), MotionBlur(blur_limit=3, p=0.6)], p=0.6),
        # IAAAdditiveGaussianNoise(scale=(0.01*255, 0.03*255), p=0.6),
        ShiftScaleRotate(0.05, 0.05, 5, always_apply=True),
    ])
    if mode != 'train' and cfg.tta:
        # def test_time_aug(image):
        # list_trans = [
        #     torch.Tensor(transform(image, cfg)).float(),
        #     torch.Tensor(transform(HorizontalFlip(p=1)(
        #         image=image)['image'], cfg)).float(),
        # ]
        #     return torch.stack(list_trans, dim=0)
        # transforms_aug = Lambda(test_time_aug)
        transforms_aug =test_time_aug
    # if cfg.cutmix:
    #     collator = CutMixCollator(cfg.cutmix_alpha)
    # else:
    #     collator = torch.utils.data.dataloader.default_collate

    pediatric_dataset = Pediatric_dicom_cutmix(label_path, data_dir, mode=mode, cfg=cfg,
                                        transforms=transforms_aug, type=type, dicom=dicom)

    # if cfg.distributed and mode == 'train':
    #     print('yes')
    #     sampler = DistributedSampler(pediatric_dataset)
    #     loader = DataLoader(
    #         pediatric_dataset, batch_size=cfg.batch_size, num_workers=4, sampler=sampler
    #     )
    # else:
    if mode == 'train':
        loader = DataLoader(pediatric_dataset,# collate_fn=collator, 
                            batch_size=cfg.batch_size, shuffle=True, num_workers=4
                            )
    else:
        loader = DataLoader(pediatric_dataset,
                            batch_size=cfg.batch_size, shuffle=False, num_workers=4
                            )
    return loader
