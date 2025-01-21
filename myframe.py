# 检测的接口函数

import cv2
import mydetect     #yolo检测
import myfatigue    #疲劳检测
import time

cap = cv2.VideoCapture(0)

def process(frame):
    frame = cv2.resize(frame, (640, 480))
    ret = []
    labellist = []
    tstart = time.time()

    # 疲劳检测
    frame, eye, mouth = myfatigue.detfatigue(frame)
    
    # YOLO检测
    action = mydetect.predict(frame)

    # 处理检测结果
    for label, prob, xyxy in action:
        labellist.append(label)
        text = f"{label} {prob:.2f}"
        left, top, right, bottom = map(int, xyxy)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.putText(frame, text, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    # 返回处理结果
    return frame, eye, mouth, labellist