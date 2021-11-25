# outside lib import
import os, sys
import argparse
import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn

# own lib import
import modules
from Classification.modeling.model import Model
from Classification.Classification_dict import dict as cls_dict
from Detection.yolov5.utils_yolov5.datasets import LoadImages, LoadStreams, IMG_FORMATS
from Detection.yolov5.utils_yolov5.torch_utils import time_sync, select_device
from Detection.yolov5.utils_yolov5.general import set_logging, non_max_suppression, xyxy2xywh
from Detection.yolov5.utils_yolov5.plots import Annotator

from Detection.eval_clothes import run_eval_clothes

from Classification.utils import utils

# TODO: print time logging
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
    net_YOLACT = modules.config_Yolact(args.yolact_weight, device)
    net_YOLO, strides, name, imgsz = modules.config_Yolov5(args.yolo_weight, device)
    deepsort = modules.config_deepsort(args.deepsort_cfg)
    net_cls = Model("efficientnet-b0",
                    True,
                    len(cls_dict["Type"]),
                    len(cls_dict["Color"]))

    # Load data
    # Re-use yolov5 data loading pipeline for simplicity
    webcam = args.source.isnumeric()
    is_img = True if any(ext in args.source for ext in IMG_FORMATS) else False

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(args.source, img_size=imgsz, stride=strides) # (sources, letterbox_img: np, orig_img: cv2, None)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(args.source, img_size=imgsz, stride=strides) # (path, letterbox_img: np, orig_img: cv2, cap)
        bs = 1

    # Run Inference
    for path, im, im0s, vid_cap in dataset:
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
        yolo_preds = net_YOLO(im_yolo)[0] # (batch, (bbox, conf, class)) type torch.Tensor
        t3 = time_sync()
        # time logging for yolo predicting
        dt1 = t3 - t2
        # nms for yolo
        yolo_preds = non_max_suppression(yolo_preds, args.yolo_conf_thres, args.yolo_iou_thres, humans, max_det=args.yolo_maxdets)
        t4 = time_sync()
        # nms time for yolo
        dt2 = t4 - t3
        # -----------------------------------------

        # yolact inference
        # -----------------------------------------
        # TODO: re-write the run_eval_clothes function, drop FastBaseTransform, drop prep_display
        im_yolact = im0s.copy() # copy to another image so we can draw on im0s later
        # type torch.Tensor, shape (batch, (bbox, conf, cls))
        # type int if no detection
        yolact_preds = run_eval_clothes(net_YOLACT,
                                        search_clothes=clothes,
                                        img_numpy=im_yolact)
        t5 = time_sync()
        # inference time for YOLACT
        dt3 = t5 - t4
        # -----------------------------------------

        # TODO: shorten the entire things
        # deepsort
        # -----------------------------------------
        # A list of objects satisfying 2 properties
        list_det = []  # list of torch.Tensor containing bbox of human
        if type(yolact_preds) == torch.Tensor and type(yolo_preds) == torch.Tensor:
            yolact_preds = yolact_preds.cpu().numpy()
            yolo_preds = yolo_preds.cpu().data.numpy()

            # Calculate inters set A and B
            def inters(bbox_a, bbox_b):
                # determine the coordinates of the intersection rectangle
                x_left = max(bbox_a[0], bbox_b[0])
                y_left = max(bbox_a[1], bbox_b[1])
                x_right = min(bbox_a[2], bbox_b[2])
                y_right = min(bbox_a[3], bbox_b[3])
                if (x_right - x_left) * (y_right - y_left) >= 0:
                    return (x_right - x_left) * (y_right - y_left)
                else:
                    return 0

            # Count = length of clothes: Draw bbox.
            # Count = not length of clothes: Not Draw bbox.
            for i in range(yolo_preds.shape[0]):
                count = 0
                for j in range(yolact_preds.shape[0]):
                    # Calculate area.
                    area_j = (yolact_preds[j][2] - yolact_preds[j][0]) * (
                                yolact_preds[j][3] - yolact_preds[j][1])
                    area = inters(yolo_preds[i, :4], yolact_preds[j, :4])
                    # Conditional
                    if area / area_j > 0.7:
                        count += 1

                if count == len(args.clothes):
                    list_det.append(np.array(yolo_preds[i, :], dtype=np.float_).tolist())

            # If length of list_det not equal 0 -> use deepsort
            if len(list_det) != 0:
                list_det = np.array(list_det)
                det = torch.from_numpy(list_det)
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(),
                                          im0s)  # np array (detections, [xyxy, track_id, class_id])
                # draw boxes for visualization
                # img_numpy = display_video(img_numpy, outputs, color=COLOR[0], search=search)
            else:
                deepsort.increment_ages()

        # classification
        # -----------------------------------------
        # 1. Read every single clothes ROI from yolact output one by one
        # 2. Perform preprocess to ROI
        # 3. Perform forward pass on image
        # 4. Convert output from classification model to correct read-able format
        # 5. Draw bbox with type and color label
        # TODO: perform forward pass on batch of images instead of single image
        if isinstance(yolact_preds, torch.Tensor):
            bboxes = yolact_preds[:, :4]  # np.ndarray
            for bbox in bboxes:
                roi = im0s[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cls_input = net_cls.preprocess(roi).to(device)
                cls_output = net_cls(cls_input)  # list (type, color)
                # type_pred: string
                # color_pred: list(string)
                type_pred, color_pred = utils.convert_output(cls_dict, cls_output)
                s = f"{type_pred}, {color_pred}"
                annotator.box_label(bbox, label=s, color=(0, 0, 255))
        # -----------------------------------------

        # show image
        cv2.imshow("A", im0s)
        if is_img:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--yolact_weight', type=str, default="Detection/train_models/yolact_base_clothes_1_30000.pth")
    parser.add_argument('--yolo_weight', type=str, default="Detection/train_models/yolo5s.pt")
    parser.add_argument('--deepsort_cfg', type=str, default="Detection/train_models/deep_sort.yaml")
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--yolo_conf_thres', '--yct', type=float, default=0.4)
    parser.add_argument('--yolo_iou_thres', '--yit', type=float, default=0.5)
    parser.add_argument('--yolo_maxdets', '--ymd', type=int, default=100)
    parser.add_argument('--humans', type=str, default="male")
    parser.add_argument('--clothes', type=str, default="short_sleeved_shirt")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)







