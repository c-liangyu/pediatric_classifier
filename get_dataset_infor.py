import os
import shutil
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm

def correct_path(path): return path if os.path.isfile(
    path) else path[:-3]+'dicom'

def iserror(func, *args, **kw):
    try:
        func(*args, **kw)
        return False
    except Exception:
        return True

def read_xray(path):

    dicom = pydicom.read_file(path)
    return dicom
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


# jpeg_dir = '/home/tungthanhlee/thanhtt/assigned_jpeg'
dicom_dir = '/home/tungthanhlee/thanhtt/assigned dicom'

# img_names = next(os.walk(jpeg_dir))[2]
# img_ids = ['.'.join(aa.split('.')[:-1]) for aa in img_names]

train_path = 'annotations_cls_2203/train.csv'
val_path = 'annotations_cls_2203/val.csv'
test_path = 'annotations_cls_2203/test.csv'
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)
df = pd.concat([train_df, val_df, test_df], ignore_index=True)
img_ids = test_df.image_id.unique()

infor_dict = dict.fromkeys(['image_id', 'age', 'sex', 'height', 'width'], 0.0)
attr_dict = {
    'image_id': 'SOPInstanceUID',
    'age': 'PatientAge',
    'sex': 'PatientSex',
    'height': 'Rows',
    'width': 'Columns'
}

for i, img_id in enumerate(tqdm(img_ids)):
    img_path = correct_path(os.path.join(dicom_dir, img_id+'.dcm'))
    dicom_content = read_xray(img_path)
    for key in infor_dict.keys():
        if i == 0:
            if hasattr(dicom_content, attr_dict[key]):
                infor_dict[key] = [getattr(dicom_content, attr_dict[key])]
            else:
                infor_dict[key] = ['']
        else:
            if hasattr(dicom_content, attr_dict[key]):
                # print(getattr(dicom_content, attr_dict[key]))
                infor_dict[key].append(getattr(dicom_content, attr_dict[key]))
            else:
                infor_dict[key].append('')

df = pd.DataFrame.from_dict(infor_dict)
df.to_csv('test_dataset_infor.csv', index=False)
    # print(dicom_content)
    # # print(dicom_content.PatientAge)
    # if i == 0:
    #     break