# general import
import torch
import torch.backends.cudnn as cudnn

import os


# YOLACT
# ------------------------------------------------------------
from yolact.yolact import Yolact
from yolact.data import set_cfg
from yolact.utils.functions import SavePath

def config_Yolact(yolact_weight):
    # Load config from weight
    model_path = SavePath.from_str(yolact_weight)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % config)
    set_cfg(config)

    with torch.no_grad():
        if not os.path.exists('Detection/results'):
            os.makedirs('results')

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
        return net
# ------------------------------------------------------------



# YOLOv5
# ------------------------------------------------------------
from yolov5.models.experimental import attempt_load
from yolov5.utils_yolov5.general import set_logging, check_img_size
from yolov5.utils_yolov5.torch_utils import select_device

def config_Yolov5(yolo_weight, imgsz=640):
    half = False
    # Initialize

    set_logging()
    device = select_device('')
    half &= device.type != 'cpu'  # half precision only supported on CUDA

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