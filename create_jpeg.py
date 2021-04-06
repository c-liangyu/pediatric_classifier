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

def iserror(func, *args, **kw):
    try:
        func(*args, **kw)
        return False
    except Exception:
        return True

def filter_df(df):
    # Filter unreadable DICOM files and duplicates

    df.drop_duplicates(inplace=True)
    img_ids = df.image_id.unique()
    rand_ids = np.random.permutation(img_ids.shape[0])
    correct_path = lambda path: path if os.path.isfile(path) else path[:-3]+'dicom'
    count = 0
    error_ids = []
    for i in tqdm(range(img_ids.shape[0])):
        path = correct_path(os.path.join('/media/tungthanhlee/DATA/tienthanh/assigned dicom', img_ids[rand_ids[i]]+'.dcm'))
        if iserror(read_xray, path):
            count += 1
            error_ids.append(img_ids[rand_ids[i]])

    df = df[df['image_id'].map(lambda a: a in error_ids) == False]

    return df

data_dir = '/media/tungthanhlee/DATA/tienthanh/assigned dicom'

def correct_path(path): return path if os.path.isfile(
    path) else path[:-3]+'dicom'

img_names = next(os.walk(data_dir))[2]
# csv_path = 'annotations_cls_2403.csv'

jpeg_dir = '/media/tungthanhlee/DATA/tienthanh/assigned_jpeg'
add_dir = '/media/tungthanhlee/DATA/tienthanh/additional_jpeg'
os.makedirs(jpeg_dir, exist_ok=True)
os.makedirs(add_dir, exist_ok=True)

for i, img_name in enumerate(tqdm(img_names)):
    path = os.path.join(data_dir, img_name)
    save_name = '.'.join(img_name.split('.')[:-1]) + '.jpg'
    save_path = os.path.join(jpeg_dir, save_name)
    add_path = os.path.join(add_dir, save_name)
    if not os.path.exists(save_path):
        if not iserror(read_xray, path):
            img = read_xray(path)
            cv2.imwrite(save_path, img)
            cv2.imwrite(add_path, img)
# modes = ['train', 'val', 'test']

# for mode in modes:
# df = pd.read_csv(csv_path)
# df = filter_df(df)
# img_ids = df.image_id.unique()
# for i, img_id in enumerate(tqdm(img_ids)):
#     img_path = os.path.join(data_dir, img_id+'.dcm')
#     img_path = correct_path(img_path)
#     img = read_xray(img_path)
#     save_path = os.path.join(jpeg_dir, img_id+'.jpg')
#     cv2.imwrite(save_path, img)