import os
import random
import sys
from collections import Counter
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from net import OCR
import collections
import itertools
import cv2
from tqdm import tqdm


def get_letters(dt):
    CHAR_VECTOR = sorted(list(Counter([name for file in dt.class_name.values for name in file.split('.')[0].strip().replace(' ', '')]).keys()))
    letters = [letter for letter in CHAR_VECTOR]
    return letters


df = pd.read_csv('images/annotations.csv')
df['image_name'] = df['image_name'].apply(lambda f: 'images/' + f)
letters = get_letters(df)
num_classes = len(letters) + 1


def decode(out, letters):
    outstr = ""
    out_best = [k for k, g in itertools.groupby(out.numpy())]
    for c in out_best:
        if c != 0:
            outstr += letters[c-1]
    return outstr



# load model
args = [64, "M", 128, "M", 256, "M", 512, "M", 512]
model = OCR(features=args, in_channels=1, num_classes=num_classes, out_features=32, num_layers=2)
model.load_state_dict(torch.load("snapshots/ocr.pth"))
model.eval()


def preprocess_input(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image -= np.amin(image)
    image /= np.amax(image)
    return image


def prepare_data(img_path: str):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (128, 64))
    image = preprocess_input(image)
    image = np.expand_dims(image, -1)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image


df = df.sample(frac = 0.1)
dt = []
for idx, col in tqdm(df.iterrows(), total=df.shape[0]):
    image = prepare_data(col['image_name'])
    out = model(image)
    text = decode(out, letters)
    dt.append(text)

df['predictions'] = dt
df['t/f'] = df.class_name == df.predictions
print(df['t/f'].value_counts(normalize=True))
df.to_csv('result.csv', index = False)
print(df)