import os, sys
import pandas as pd
from ast import literal_eval
import numpy as np
from tqdm import tqdm
import pickle
import yaml
import matplotlib
import matplotlib.pyplot as plt


import cv2
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.augmentation import Augmentation


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


class ClothesClassificationDataset(Dataset):
    def __init__(self, path, task, dict: dict, imgsz,
                 augment=False, augment_config=None):
        self.path = path
        self.task = task
        self.type_len = len(dict["Type"])
        self.color_len = len(dict["Color"])
        self.clothes_type = dict["Type"]
        self.clothes_color = dict["Color"]
        self.imgsz = imgsz
        self.augment = augment
        if self.augment and augment_config is not None:
            self.augment_config = augment_config
            self.transform = Augmentation(self.augment_config)
            self.transform.define_aug()

    def get_csv(self):
        csv_path = os.path.join(self.path, f"{self.task}.csv")
        try:
            dataframe = pd.read_csv(csv_path)
        except FileNotFoundError:
            column_name =["filename", "clothes_type",
                          "clothes_color", "type_onehot", "color_onehot"]
            value_list = []
            for folder in os.listdir(self.path):
                folder_path = os.path.join(self.path, folder)
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
        image = cv2.imread(os.path.join(self.path, image_name))
        image = image[:, :, ::-1] # BGR to RGB
        # transform
        if self.augment:
            image = self.transform(image)
        image = image.transpose(2, 0, 1) # (C,H,W)
        image = np.ascontiguousarray(image)
        type = dataframe.iloc[idx, 1]
        color = dataframe.iloc[idx, 2]
        type_onehot = torch.tensor(literal_eval(dataframe.iloc[idx, 3]), dtype=torch.float)
        color_onehot = torch.tensor(literal_eval(dataframe.iloc[idx, 4]), dtype=torch.float)
        sample = {"image_name": image_name,
                  "image": torch.from_numpy(image),
                  "type": type,
                  "color": color,
                  "type_onehot": type_onehot, #torch.Tensor
                  "color_onehot": color_onehot} #torch.Tensor
        sample["image"] = transforms.Resize((self.imgsz, self.imgsz))(sample["image"])
        return sample


    def get_color_statistic(self):
        dataframe = self.get_csv()
        cache_path = os.path.join(self.path, "color_statistic.cache")
        if not os.path.exists(cache_path):
            num_color_dict = {}
            for i in range(self.color_len):
                var = f'{self.clothes_color[i]}'
                num_color_dict[var] = 0
            color_series = dataframe["clothes_color"]
            for colors in color_series:
                colors = colors.split('_')
                for color in colors:
                    color = color.lower()
                    num_color_dict[f'{color}'] += 1
            with open(cache_path, 'wb') as f:
                pickle.dump(num_color_dict, f)
        with open(cache_path, 'rb') as f:
            num_color_dict = pickle.load(f)
        return num_color_dict


    def plot_labels(self):
        print(f"Plotting labels to {self.path}/labels.jpg...")
        dataframe = self.get_csv()
        cache_path = os.path.join(self.path, "type_statistic.cache")
        if not os.path.exists(cache_path):
            # construct dict for storing clothes type
            num_type_dict = {}
            for i in range(self.type_len):
                key = f'{self.clothes_type[i]}'
                num_type_dict[key] = 0

            # count value to clothes type
            for _type in dataframe["clothes_type"]:
                num_type_dict[_type] += 1

            # save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(num_type_dict, f)

        with open(cache_path, 'rb') as f:
            num_type_dict = pickle.load(f)
        num_color_dict = self.get_color_statistic()

        matplotlib.use('svg')  # faster

        # plot types
        fig1, ax1 = plt.subplots(figsize=(20, 6))
        num_type = [v for v in num_type_dict.values()]
        types = [k for k in num_type_dict.keys()]
        ax1.set_xticks(range(len(types)))
        ax1.bar(types, num_type)
        plt.savefig(f"{self.path}/types_label.jpg")

        # plot color
        fig2, ax2 = plt.subplots(figsize=(20, 6))
        num_color = [v for v in num_color_dict.values()]
        colors = [k for k in num_color_dict.keys()]
        ax2.set_xticks(range(len(colors)))
        ax2.bar(colors, num_color)
        plt.savefig(f"{self.path}/colors_label.jpg")

        matplotlib.use('Agg')
        plt.show()
        plt.close()


def create_dataloader(dataset, imgsz, batch_size, workers, task='train', augment=False, augment_config=None):
    if not isinstance(dataset, dict):
        with open(dataset) as f:
            data_dict = yaml.safe_load(f)
    else:
        data_dict = dataset

    path = os.path.join(data_dict['root'], task)
    cls_dict: dict = data_dict['class']
    dataset = ClothesClassificationDataset(path, task, cls_dict, imgsz, augment, augment_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return dataloader, dataset
