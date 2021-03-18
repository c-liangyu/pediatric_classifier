import pandas as pd
import numpy as np
import os
import shutil

csv_path = '/media/tungthanhlee/DATA/tung/ChestXrayPackage/ChestXray/src/Chexpert_Model_Building/data/csv/train_chexmic.csv'
# csv_path = '/media/tungthanhlee/DATA/tung/ChestXrayPackage/ChestXray/src/Chexpert_Model_Building/data/csv/valid_chexpert.csv'
data_dir = '/home/tungthanhlee/bdi_xray/data/images'

df = pd.read_csv(csv_path)
print(len(df))

print(df.head())
print(list(df.columns))
df = pd.read_csv('annotations_cls_1503/train.csv')
print(len(df), sum(df['No finding'].values))