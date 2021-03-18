from genericpath import exists
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm
import os

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# train_df = pd.read_csv('train.csv')
# val_df = pd.read_csv('val.csv')
# test_df = pd.read_csv('holdout.csv')
# print(len(train_df), len(val_df), len(test_df))

# df = pd.concat([train_df, val_df, test_df], ignore_index=True)

csv_path = 'annotations_cls_1803.csv'
df = pd.read_csv(csv_path)

def iserror(func, *args, **kw):
    try:
        func(*args, **kw)
        return False
    except Exception:
        return True

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

def Iterative_Stratifier_Split(df, ratio=0.15):
    img_ids = df.image_id.unique()
    print("Creating one-hot labels ...")
    labels = np.zeros((len(img_ids), 13), dtype=np.uint8)
    for i, img_id in enumerate(tqdm(img_ids)):
    # for i, img_id in enumerate(img_ids):
        aa = df.loc[df.image_id == img_id, :].to_numpy()[0, 1:]
        # print(aa.shape)
        labels[i] = aa
    print("Done!")
    print("Spliting dataset ...")
    train_image_id, train_class_id, test_image_id, test_class_id = iterative_train_test_split(img_ids.reshape(-1,1), labels, test_size=ratio)
    train_df = df[df.image_id.map(lambda x: x in train_image_id.reshape(-1))]
    test_df = df[df.image_id.map(lambda x: x in test_image_id.reshape(-1))]
    print("Done!\n")
    return train_df, test_df

# filter data frame

cur_df = filter_df(df)
# cur_df = df
# cur_df = df[df.class_id!=7].reset_index(drop = True)
# print(len(df), len(cur_df))

# split 7 classes with abnormalities

test_ratio = 0.15
val_ratio = 0.15
train_ratio = 0.7

train_df, test_df = Iterative_Stratifier_Split(cur_df, test_ratio)
train_df, val_df = Iterative_Stratifier_Split(train_df, val_ratio/(val_ratio+train_ratio))

# split no finding class

# nf_df = df[df.class_id == 7]
# nf_img_ids = nf_df.image_id.unique()
# nf_len = nf_img_ids.shape[0]

# rand_id = np.random.permutation(len(nf_img_ids))

# nf_test_ids = nf_img_ids[rand_id[:int(nf_len*test_ratio)]]
# nf_val_ids = nf_img_ids[rand_id[int(nf_len*test_ratio):int(nf_len*val_ratio)+int(nf_len*test_ratio)]]
# nf_train_ids = nf_img_ids[rand_id[int(nf_len*val_ratio)+int(nf_len*test_ratio):]]

# nf_test_df = df[df.image_id.map(lambda x: x in nf_test_ids)]
# nf_val_df = df[df.image_id.map(lambda x: x in nf_val_ids)]
# nf_train_df = df[df.image_id.map(lambda x: x in nf_train_ids)]

# concatnate all classes

# train_df = pd.concat([train_df, nf_train_df], ignore_index=True)
# val_df = pd.concat([val_df, nf_val_df], ignore_index=True)
# test_df = pd.concat([test_df, nf_test_df], ignore_index=True)

print(len(train_df), len(val_df), len(test_df))

output_dir = os.path.basename(csv_path).split('.')[0]
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir,'train.csv'), index=False)
val_df.to_csv(os.path.join(output_dir,'val.csv'), index=False)
test_df.to_csv(os.path.join(output_dir,'test.csv'), index=False)