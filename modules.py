# general import
import torch
import torch.backends.cudnn as cudnn

import os, sys
sys.path.insert(0, "Detection")


# YOLACT
# ------------------------------------------------------------
from yolact.yolact import Yolact
from yolact.data import set_cfg
from yolact.utils.functions import SavePath


# TODO: make config as a parameter instead of using a global parameter from yolact.data
def config_Yolact(yolact_weight, device):
    # Load config from weight
    print("Loading YOLACT" + '-'*10)
    model_path = SavePath.from_str(yolact_weight)
    config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % config)
    set_cfg(config)

    with torch.no_grad():
        # Temporarily disable to check behavior
        # Behavior: disabling this cause torch.Tensor(list, device='cuda') not working
        # Currently enable for now
        # TODO: Will find a workaround to disable this behavior
        # Use cuda
        use_cuda = True
        if use_cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Eval for image, images or video
        net = Yolact()
        net.load_weights(yolact_weight)
        net.eval()
        print("Done loading YOLACT" + '-'*10)
        return net.cuda()
# ------------------------------------------------------------



# YOLOv5
# ------------------------------------------------------------
from yolov5.models.experimental import attempt_load
from yolov5.utils_yolov5.general import check_img_size

def config_Yolov5(yolo_weight, device, imgsz=416):
    # Load model
    model = attempt_load(yolo_weight, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    imgsz = check_img_size(imgsz, s=stride)

    return model, stride, names, imgsz

# ------------------------------------------------------------



# Deepsort
# ------------------------------------------------------------
from deep_sort.deep_sort_pytorch.utils.parser import get_config
from deep_sort.deep_sort_pytorch.deep_sort import DeepSort

def config_deepsort(deepsort_cfg):
# initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(deepsort_cfg)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                      max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                      max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                      use_cuda=True)
    return deepsort

# ------------------------------------------------------------



# Classification
# ------------------------------------------------------------
# TODO: create a yaml config file to load classification model from
from Classification.modeling.model import Model

def config_clsmodel(clsmodel_config):
    pass

# ------------------------------------------------------------