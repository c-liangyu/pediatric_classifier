from numpy.core.numeric import NaN
import pandas as pd
import os
import numpy as np
import statistics


def filter_row(row):
    age = row.age
    
    if age[-1] == 'Y' and len(age) > 1:
        age = int(age[:-1])
        if age > 10:
            row.age = NaN
        else:
            row.age = age
    elif len(age)==1:
        row.age = 1.0
    else:
        row.age = 0
    
    sex = row.sex
    if sex == 'O':
        row.sex = NaN
    
    return row
    

csv_path = 'test_dataset_infor.csv'
df = pd.read_csv(csv_path)
df.dropna(inplace=True)
list_unique_age = dict.fromkeys(df.age.unique(), 0.0)
for key in list_unique_age.keys():
    list_unique_age[key] = len(df[df.age == key])
df = df.apply(filter_row, axis=1)
df.dropna(inplace=True)
print(df.head(n=5))
print(np.mean(df.age))
print(len(df[df.sex == 'M'])/ len(df))
print(len(df[df.sex == 'F'])/ len(df))
print(np.mean(df.height), np.max(df.height), np.min(df.height))
print(np.mean(df.width), np.max(df.width), np.min(df.width))