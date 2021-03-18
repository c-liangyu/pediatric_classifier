import pandas as pd
import torch
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Resize, HorizontalFlip, RandomBrightnessContrast, OneOf, Blur, MotionBlur, IAAAdditiveGaussianNoise, ShiftScaleRotate
from albumentations.pytorch import ToTensor
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data.utils import transform


def preprocess(img_size): return Compose(
    [Resize(img_size[0], img_size[1]), ToTensor()])


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
        self.dicom = dicom
        self.df = pd.read_csv(label_path)
        if self.type == 'chexmic':
            self.img_ids = self.df.Path.unique()
        else:
            self.img_ids = self.df.image_id.unique()
        self.transforms = transforms
        self.mode = mode
        self.cfg = cfg
        if self.type == 'pediatric':
            self.disease_classes = ["Other opacity", "Reticulonodular opacity", "Peribronchovascular interstitial opacity", "Diffuse aveolar opacity", "Lung hyperinflation",
                                    "Consolidation", "Bronchial thickening", "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Pneumonia"]
        elif self.type == 'chexmic':
            self.disease_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                                    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        else:
            self.disease_classes = ["No finding"]
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
                    self.data_dir, self.mode, img_id+'.jpg')
            # print(img_path)
            image = cv2.imread(img_path, 0)
        if self.type == 'pediatric':
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

        if self.mode == 'train' and self.transforms is not None:
            image = self.transforms(image=image)['image']
        image = transform(image, self.cfg)
        image = torch.Tensor(image).float()
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # img_size = (self.cfg.long_side, self.cfg.long_side)
        # image = preprocess(img_size)(image=image)['image']
        # print(type(image), type(label))
        # print(image.shape, label.shape)

        return image, label


def create_loader(label_path, data_dir, cfg, mode='train', dicom=False, type=None):

    transforms = Compose([
        HorizontalFlip(),
        # RandomBrightnessContrast(
        #     always_apply=True, brightness_limit=0.2, contrast_limit=0.2),
        # OneOf([Blur(blur_limit=2, p=0.6), MotionBlur(blur_limit=3, p=0.6)], p=0.6),
        # IAAAdditiveGaussianNoise(scale=(0.01*255, 0.03*255), p=0.6),
        ShiftScaleRotate(0.05, 0.05, 5, always_apply=True),
    ])
    if cfg.distributed:
        pediatric_dataset = Pediatric_dicom(label_path, data_dir, mode=mode, cfg=cfg,
                                            transforms=transforms, type=type, dicom=dicom)
        sampler = DistributedSampler(pediatric_dataset)
        loader = DataLoader(
            pediatric_dataset, batch_size=cfg.batch_size,
            shuffle=True, num_workers=4, sampler=sampler
        )
    else:
        loader = DataLoader(
            Pediatric_dicom(label_path, data_dir, mode=mode, cfg=cfg,
                            transforms=transforms, type=type, dicom=dicom),
            batch_size=cfg.batch_size, shuffle=True, num_workers=4
        )

    return loader
