from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from Classification.utils.utils import get_imgsz


class Model(nn.Module):
    def __init__(self, base_model: str,
                 use_pretrained:bool,
                 num_class_1,
                 num_class_2,
                 dropout_rate=0.2):
        super(Model, self).__init__()
        if use_pretrained:
            self.base_model = EfficientNet.from_pretrained(base_model, include_top=False)
        else:
            self.base_model = EfficientNet.from_name(base_model, include_top=False)
        self.out_channel = utils.round_filters(1280, self.base_model._global_params)
        self._type_cls = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.out_channel, num_class_1)
        )
        self._color_cls = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.out_channel, num_class_2)
        )
        self.imgsz = self.base_model.get_image_size(base_model)

    def forward(self, inputs):
        """ Perform forward pass on the input """
        x = self.base_model.extract_features(inputs)
        x = self.base_model._avg_pooling(x)
        x = x.flatten(start_dim=1)

        type = self._type_cls(x)
        color = torch.sigmoid(self._color_cls(x))

        return [type, color]

    def preprocess(self, imgs):
        """ Pre-process any given image or image batch to have desired input shape """
        #imgsz = get_imgsz(self.base_model)
        if isinstance(imgs, np.ndarray):
            if imgs.ndim != 4:
                imgs = np.expand_dims(imgs, axis=0) # b, h, w, c
            imgs = imgs.transpose((0, 3, 1, 2)) # to b, c, h, w
            imgs = torch.from_numpy(imgs) # torch.Tensor
        # sanity check
        assert isinstance(imgs, torch.Tensor), "input must be a numpy array or torch Tensor"
        assert imgs.dim() == 4, "Tensor must have shape (b, c, h, w)"

        imgs = transforms.Resize((self.imgsz, self.imgsz))(imgs)
        imgs = imgs / 255.
        return imgs


