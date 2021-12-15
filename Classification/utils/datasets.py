import os, sys
import pandas as pd
import cv2
from ast import literal_eval
import numpy as np
from tqdm import tqdm
import pickle

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.augumentation import Augmentation


class ClothesClassificationDataset(Dataset):
    def __init__(self, root_dir, csv_file, dict: dict, imgsz, augment=False, augment_config=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.type_len = len(dict["Type"])
        self.color_len = len(dict["Color"])
        self.clothes_type = dict["Type"]
        self.clothes_color = dict["Color"]
        self.imgsz = imgsz
        self.augment = augment
        if self.augment:
            self.augment_config = augment_config
            self.transform = Augmentation(self.augment_config)
            self.transform.define_aug()

    def get_csv(self):
        csv_path = os.path.join(self.root_dir, self.csv_file)
        try:
            dataframe = pd.read_csv(csv_path)
        except FileNotFoundError:
            column_name =["filename", "clothes_type",
                          "clothes_color", "type_onehot", "color_onehot"]
            value_list = []
            for folder in os.listdir(self.root_dir):
                folder_path = os.path.join(self.root_dir, folder)
                for image_name in tqdm(os.listdir(folder_path), desc=f'{folder}'):
                    string = image_name.split("__")
                    type = string[0]
                    colors = string[1]
                    try:
                        type_idx = torch.tensor(self.clothes_type.index(type))
                        type_onehot = torch.nn.functional.one_hot(type_idx, num_classes=self.type_len)
                        color_onehot = torch.zeros(self.color_len, dtype=torch.int)
                        clrs = colors.split("_")
                        for color in clrs:
                            color = color.lower()
                            color_idx = self.clothes_color.index(color)
                            color_onehot[color_idx] = 1
                        value = (folder + '/' + image_name,
                                 type,
                                 colors,
                                 type_onehot.tolist(),
                                 color_onehot.tolist())
                        value_list.append(value)
                    except ValueError as e:
                        with open("Err.txt", 'a') as f:
                            f.write(folder + '/' + image_name + "\n")
                        pass
            dataframe = pd.DataFrame(value_list, columns=column_name)
            dataframe.to_csv(csv_path, index=None)
            print('Successfully created the CSV file: {}'.format(csv_path))
            dataframe = pd.read_csv(csv_path)
        return dataframe


    def __len__(self):
        dataframe = self.get_csv()
        return len(dataframe)


    def __getitem__(self, idx):
        dataframe = self.get_csv()
        image_name = dataframe.iloc[idx, 0]
        image = cv2.imread(os.path.join(self.root_dir, image_name))
        image = image[:, :, ::-1] # BGR to RGB
        # transform
        if self.augment:
            image = self.transform(image)
        image = image.transpose(2, 0, 1) # (C,H,W)
        image = np.ascontiguousarray(image) # for faster accessing
        type = dataframe.iloc[idx, 1]
        color = dataframe.iloc[idx, 2]
        type_onehot = torch.tensor(literal_eval(dataframe.iloc[idx, 3]), dtype=torch.float)
        color_onehot = torch.tensor(literal_eval(dataframe.iloc[idx, 4]), dtype=torch.float)
        sample = {"image_name": image_name,
                  "image": torch.from_numpy(image),
                  "type": type,
                  "color": color,
                  "type_onehot": type_onehot,
                  "color_onehot": color_onehot}
        sample["image"] = transforms.Resize((self.imgsz, self.imgsz))(sample["image"])
        return sample


    def get_statistic(self):
        dataframe = self.get_csv()
        cache_path = os.path.join(self.root_dir, "statistic.cache")
        if not os.path.exists(cache_path):
            num_color_dict = {}
            for i in range(self.color_len):
                var = f'num_{self.clothes_color[i]}'
                num_color_dict[var] = 0
            color_series = dataframe["clothes_color"]
            for colors in color_series:
                colors = colors.split('_')
                for color in colors:
                    color = color.lower()
                    num_color_dict[f'num_{color}'] += 1
            with open(os.path.join(self.root_dir, "statistic.cache"), 'wb') as f:
                pickle.dump(num_color_dict, f)
        print("Loading data statistic from statistic.cache")
        with open(os.path.join(self.root_dir, "statistic.cache"), 'rb') as f:
            num_color_dict = pickle.load(f)
        return num_color_dict



def create_dataloader(root_dir, csv_file, cls_dict, image_size, batch_size,
                      augment=False, augment_config=None, workers=2):
    dataset = ClothesClassificationDataset(root_dir, csv_file, cls_dict, image_size, augment, augment_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return dataloader, dataset

