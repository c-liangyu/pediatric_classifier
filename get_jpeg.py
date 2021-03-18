import pydicom
import os
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import pandas as pd
from tqdm import tqdm

def read_xray(path):

    dicom = pydicom.read_file(path)
    # print(dicom.RescaleIntercept, dicom.RescaleSlope)
    # VOI LUT is used to transform raw DICOM data to "human-friendly" view

    ######### TEST #############
    slope = dicom.RescaleSlope
    intercept = dicom.RescaleIntercept
    img = dicom.pixel_array * slope + intercept
    data = apply_voi_lut(img, dicom)
    ############################

    # data = apply_voi_lut(dicom.pixel_array, dicom)

    # depending on this value, X-ray may look inverted - fix that:
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data

data_dir = '/media/tungthanhlee/DATA/tienthanh/assigned dicom'

def correct_path(path): return path if os.path.isfile(
    path) else path[:-3]+'dicom'

csv_dict = {'train': 'annotations_cls_1803/train.csv',
            'val':'annotations_cls_1803/val.csv',
            'test':'annotations_cls_1803/test.csv'}

jpeg_dir = 'annotations_cls_1803/images'
os.makedirs(jpeg_dir, exist_ok=True)
modes = ['train', 'val', 'test']

for mode in modes:
    os.makedirs(os.path.join(jpeg_dir, mode), exist_ok=True)
    df = pd.read_csv(csv_dict[mode])
    img_ids = df.image_id.unique()
    for i, img_id in enumerate(tqdm(img_ids)):
        img_path = os.path.join(data_dir, img_id+'.dcm')
        img_path = correct_path(img_path)
        img = read_xray(img_path)
        save_path = os.path.join(jpeg_dir, mode, img_id+'.jpg')
        cv2.imwrite(save_path, img)