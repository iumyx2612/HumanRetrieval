# outside lib import
import os, sys
import argparse
import numpy as np
import cv2
import shutil

import torch
import torch.backends.cudnn as cudnn

# own lib import
import modules
from Classification.modeling.model import Model
from Classification.Classification_dict import dict as cls_dict
from Detection.yolov5.utils.datasets import LoadImages, LoadStreams, IMG_FORMATS
from Detection.yolov5.utils.torch_utils import time_sync, select_device
from Detection.yolov5.utils.general import set_logging, non_max_suppression, xyxy2xywh, scale_coords
from Detection.yolov5.utils.plots import Annotator

from Detection.eval_clothes import run_eval_clothes

from Detection.deep_sort.deep_sort_pytorch.utils.draw import draw_boxes

from Classification.utils import utils


@torch.no_grad()
def run(args):
    # Initialize
    set_logging()
    device = select_device(args.device)

    # TODO: map with config data instead of string processing
    humans = args.humans
    if len(humans) == 1:
        humans = [humans.index("".join(humans))]
    else:
        humans = [0, 1]  # With 0 is male and 1 is female.

    if ',' in args.clothes:
        clothes = args.clothes.split(',')
    else:
        clothes = [args.clothes]
    # disable for now
    '''if not all(elem in class_clothes for elem in search_clothes):
        raise ValueError(f"Have any category not exist in parameter classes")'''

    # Load nets
    net_YOLO1, strides1, yolo_name1, imgsz1 = modules.config_Yolov5(args.yolo_weight, device)
    net_YOLO2, strides2, yolo_name2, imgsz2 = modules.config_Yolov5(args.yolact_weight, device)
    #deepsort = modules.config_deepsort(args.deepsort_cfg)
    net_cls = Model('efficientnet-b0',
                  use_pretrained=False,
                  num_class_1=12,
                  num_class_2=9)
    net_cls.load_state_dict(torch.load(args.cls_weight)['state_dict'])
    net_cls.to(device)
    net_cls.eval()

    # Load data
    # Re-use yolov5 data loading pipeline for simplicity
    webcam = args.source.isnumeric()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(args.source, img_size=imgsz1, stride=strides1) # (sources, letterbox_img: np, orig_img: cv2, None)
    else:
        dataset = LoadImages(args.source, img_size=imgsz1, stride=strides1) # (path, letterbox_img: np, orig_img: cv2, cap)

    cv2.namedWindow("A", cv2.WINDOW_FREERATIO)
    cv2.resizeWindow("A", 1280, 720)
    # saving prediction video
    if args.savevid:
        width = next(iter(dataset))[3].get(cv2.CAP_PROP_FRAME_WIDTH)
        height = next(iter(dataset))[3].get(cv2.CAP_PROP_FRAME_HEIGHT)
        res = (int(width), int(height))
        # this format fail to play in Chrome/Win10/Colab
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
        # fourcc = cv2.VideoWriter_fourcc(*'H264') #codec
        output = cv2.VideoWriter(args.savename, fourcc, 30, res)


    # Run Inference
    counter = 0
    for path, im, im0s, vid_cap, _ in dataset:
        human_label = ""
        counter += 1
        is_img = True if any(ext in path for ext in IMG_FORMATS) else False
        annotator = Annotator(np.ascontiguousarray(im0s),
                              line_width=2,
                              font_size=1)
        # yolo inference
        # -----------------------------------------
        t1 = time_sync()
        im_yolo = torch.from_numpy(im).to(device) # yolo input
        im_yolo = im_yolo.float()
        im_yolo /= 255
        if len(im_yolo.shape) == 3:
            im_yolo = im_yolo[None]  # expand for batch dim
        t2 = time_sync()
        # time logging for data loading
        dt0 = t2 - t1
        # Inference on yolov5
        yolo_preds = net_YOLO1(im_yolo) # (batch, (bbox, conf, class)) type torch.Tensor
        t3 = time_sync()
        # time logging for yolo predicting
        dt1 = t3 - t2
        # nms for yolo
        # yolo_preds: torch.Tensor
        yolo_preds = non_max_suppression(yolo_preds, args.yolo_conf_thres, args.yolo_iou_thres, humans, max_det=args.yolo_maxdets)[0]
        t4 = time_sync()
        # nms time for yolo
        dt2 = t4 - t3

        # scale yolo preds to im0
        if len(yolo_preds):
            yolo_preds[:, :4] = scale_coords(im_yolo.shape[2:], yolo_preds[:, :4], im0s.shape).round()
            # =======================================================================
            # TODO: CHEAT CODE
            for *xyxy, conf, cls in yolo_preds:
                c = int(cls)
                human_label = yolo_name1[c]
            # =======================================================================

        # -----------------------------------------

        # yolact inference
        # -----------------------------------------
        # TODO: re-write the run_eval_clothes function, drop FastBaseTransform, drop prep_display
        clothes_preds = net_YOLO2(im_yolo)
        clothes_preds = non_max_suppression(clothes_preds, 0.4, 0.45, [0, 1])[0]
        t5 = time_sync()
        # inference time for YOLACT
        dt3 = t5 - t4
        # -----------------------------------------

        # classification
        # -----------------------------------------
        # 1. Read every single clothes ROI from yolact output one by one
        # 2. Perform preprocess to ROI
        # 3. Perform forward pass on image
        # 4. Convert output from classification model to correct read-able format
        # 5. Draw bbox with type and color label
        # TODO: perform forward pass on batch of images instead of single image
        t6 = time_sync()
        clothes_labels = []
        if len(clothes_preds):
            clothes_preds[:, :4] = scale_coords(im_yolo.shape[2:], clothes_preds[:, :4], im0s.shape).round()
            bboxes = clothes_preds[:, :4]  # np.ndarray
            for bbox in bboxes:
                roi = im0s[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cls_input = net_cls.preprocess(roi)
                cls_output = net_cls(cls_input)  # list (type, color)
                # type_pred: string
                # color_pred: list(string)
                type_pred, color_pred = utils.convert_output(cls_dict, cls_output)
                # =======================================================================
                # TODO: CHEAT CODE
                temp = '_'.join(x for x in color_pred)
                s = f"{type_pred}-{temp}"
                clothes_labels.append(s)
                # =======================================================================
                #annotator.box_label(bbox, label=s, color=(0, 0, 255))
        # =======================================================================
        # TODO: CHEAT CODE
        clothes_label = '--'.join(x for x in clothes_labels)
        l = f"{human_label}--{clothes_label}--{counter}.jpg"
        new_path = f"TestDataset/a/{l}"
        shutil.copy(path, new_path)

        t7 = time_sync()
        dt4 = t7 - t6
        # -----------------------------------------

        # show image
        if args.view_img:
            cv2.imshow("A", im0s)
            if is_img:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)

        # save preds
        if args.savevid:
            output.write(im0s)

        # time logging
        print(f"Inference time \n YOLO: {dt1} \t YOLACT: {dt3} \t Classification: 0")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--yolact_weight', type=str, default="Detection/train_models/clothes_v5s.pt")
    parser.add_argument('--yolo_weight', type=str, default="Detection/train_models/yolov5s_640.pt")
    #parser.add_argument('--deepsort_cfg', type=str, default="Detection/train_models/deep_sort.yaml")
    parser.add_argument('--cls_weight', type=str, default="Classification/weights/effnet_b0_2412.pt")
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--yolo_conf_thres', '--yct', type=float, default=0.4)
    parser.add_argument('--yolo_iou_thres', '--yit', type=float, default=0.5)
    parser.add_argument('--yolo_maxdets', '--ymd', type=int, default=100)
    parser.add_argument('--humans', type=str, default="male")
    parser.add_argument('--clothes', type=str, default="short_sleeved_shirt")
    parser.add_argument('--view_img', action="store_true")
    parser.add_argument('--savevid', action="store_true")
    parser.add_argument('--savename', type=str, default="results/out.avi")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)







