from modeling.model import Model
from Classification_dict import dict as clsdict
from utils.utils import select_device, get_imgsz, convert_output

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse



def run(args):
    device = select_device(args.device)
    imgsz = get_imgsz(args.base_model)
    model = Model(args.base_model,
                  use_pretrained=False,
                  num_class_1=args.num_cls1,
                  num_class_2=args.num_cls2)
    model.load_state_dict(torch.load(args.weights)['state_dict'])
    model.to(device)
    model.eval()

    orig_image = cv2.imread(args.image_path)[:, :, ::-1] #BGR to RGB
    image = torch.from_numpy(np.expand_dims(orig_image.transpose(2, 0, 1), axis=0).copy()).to(device) #convert to (batch, c, h, w)
    image = transforms.Resize(imgsz)(image) / 255.

    output = model(image) #list [type_preds: torch.Tensor, color_preds: torch.Tensor]

    type_pred, color_pred = convert_output(clsdict, output)
    plt.imshow(orig_image)
    plt.title(type_pred + "\n" + str(color_pred))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--base_model', default='efficientnet-b0', type=str, help='feature extractor')
    parser.add_argument('--weights', default="weights/little_3_12.pt", type=str, help='path to weights for model')
    parser.add_argument('--num_cls1', required=True, type=int, help='number of class 1')
    parser.add_argument('--num_cls2', required=True, type=int, help='number of class 2')
    parser.add_argument('--image_path', required=True, type=str, help='path to image')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)

