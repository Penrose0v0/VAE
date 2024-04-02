import torch
from torch.utils.data.dataset import Dataset
import os
import cv2
import numpy as np

from utils import normalize_image


class VAEDataset(Dataset):
    def __init__(self, image_folder="dataset", input_shape=(512, 512)):
        super(VAEDataset, self).__init__()
        self.image_folder = image_folder
        self.input_shape = input_shape
        self.data = self.read_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = np.array(self.data[item]).astype(np.float32)
        return torch.from_numpy(data)

    @property
    def read_data(self):
        # Reading
        print(f"Images: {self.image_folder}")
        image_files = os.listdir(self.image_folder)
        print("Reading data... ", end='')

        data = []
        total = 0
        for image_file in image_files:
            # Add image and preprocess
            image = cv2.imread(os.path.join(self.image_folder, image_file))
            image = cv2.resize(image, self.input_shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(normalize_image(image), (2, 0, 1))

            data.append(image)
            total += 1

        print(f"Complete! Total: {total}")
        return data



