# yolo检测的接口函数
# 详细信息请参考 https://blog.csdn.net/qq_20241587/article/details/113349874?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-6.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-6.control

import numpy as np
import cv2
import torch
from numpy import random
#import evaluator
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device, time_synchronized
 
 
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
 
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
 
 
weights = r'weights/best.pt'
# 优先使用GPU
opt_device = '0' if torch.cuda.is_available() else 'cpu'  
imgsz = 640
opt_conf_thres = 0.6
opt_iou_thres = 0.45

# Initialize
device = select_device(opt_device)
half = device.type != 'cpu'  # 在GPU上使用半精度
model = attempt_load(weights, map_location=device)
imgsz = check_img_size(imgsz, s=model.stride.max())
if half:
    model.half()

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def predict(im0s):
    if not hasattr(predict, 'img'):
        predict.img = torch.zeros((1, 3, imgsz, imgsz), device=device)
        _ = model(predict.img.half() if half else predict.img)

    # 预处理
    img = letterbox(im0s, new_shape=imgsz, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推理
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)

    # 处理结果
    ret = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in reversed(det[:5]):
                try:
                    cls_index = int(cls) % len(names)
                    if conf > opt_conf_thres:
                        label = f'{names[cls_index]}'
                        ret.append((label, float(conf), xyxy))
                except Exception as e:
                    continue
    return ret

def map_class_index(cls_index, num_classes=4):
    """将大数值的类别索引映射到有效范围内"""
    if isinstance(cls_index, torch.Tensor):
        cls_index = cls_index.item()
    return cls_index % num_classes