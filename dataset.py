import torch
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
import albumentations as A
import cv2


def preprocess_input(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image -= np.amin(image)
    image /= np.amax(image)
    return image


transforms = A.Compose([A.Rotate(limit=5, p=0.5),
                        A.Affine(shear=15, p=0.5)])


class TextImageGenerator(object):
    def __init__(self, data, letters: list, max_len: int, img_w: int = 128, img_h: int = 64, transform=False):
        self.data = data
        self.img_w = img_w
        self.img_h = img_h
        self.max_len = max_len
        self.letters = letters
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def text_to_labels(self, text: str):
        data = list(map(lambda x: self.letters.index(x) + 1, text))
        while len(data) < self.max_len:
            data.append(0)
        length = [len(text)]
        return data, length

    def prepare_data(self, img_path: str) -> torch.Tensor:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_w, self.img_h))
        if self.transform:
            image = transforms(image=image)["image"]
        image = preprocess_input(image)
        image = np.expand_dims(image, -1)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __getitem__(self, index):
        img = self.prepare_data(self.data.iloc[index, 0])
        text = self.data.iloc[index, 1]
        encode, length = self.text_to_labels(text)
        encode = torch.tensor(encode, dtype=torch.long)
        return img, [encode, torch.IntTensor(length)]


