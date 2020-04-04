import numpy as np
import pandas as pd
import os
import matplotlib
from shutil import copyfile
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook
# import cv2 as cv

DATA_FOLDER = 'dataloaders/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'
TRAIN_REAL_FAKE_FOLDER = "train"

print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
ext_dict = []
for file in train_list:
    file_ext = file.split('.')[1]
    if (file_ext not in ext_dict):
        ext_dict.append(file_ext)
print(f"Extensions: {ext_dict}")    

test_list = list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))
ext_dict = []
for file in test_list:
    file_ext = file.split('.')[1]
    if (file_ext not in ext_dict):
        ext_dict.append(file_ext)
print(f"Extensions: {ext_dict}")
for file_ext in ext_dict:
    print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")

json_file = [file for file in train_list if  file.endswith('json')][0]
print(f"JSON file: {json_file}")

def get_meta_from_json(path):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df


meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)
for idx, meta in enumerate(meta_train_df.values):
    video = meta_train_df.index[idx]
    print("Video: ", video, " label: ",meta[0])
    src = os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video)
    dst = os.path.join(DATA_FOLDER, TRAIN_REAL_FAKE_FOLDER , meta[0], video )
    copyfile(src, dst)


    