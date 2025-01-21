# 疲劳检测，检测眼睛和嘴巴的开合程度

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2
import math
import time
from threading import Thread
import os
import sys

# 初始化dlib的人脸检测器
detector = dlib.get_frontal_face_detector()

# 使用weights文件夹下的模型
model_path = os.path.join('weights', 'shape_predictor_68_face_landmarks.dat')
if not os.path.exists(model_path):
    # 如果weights文件夹下没有，尝试在根目录查找
    model_path = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(model_path):
        print("错误：找不到人脸关键点模型文件")
        print("请确保以下任一路径存在模型文件：")
        print("1. weights/shape_predictor_68_face_landmarks.dat")
        print("2. shape_predictor_68_face_landmarks.dat")
        sys.exit(1)

try:
    predictor = dlib.shape_predictor(model_path)
    print(f"成功加载人脸关键点模型: {model_path}")
except Exception as e:
    print(f"加载模型出错: {e}")
    sys.exit(1)

# 定义眼睛和嘴巴的关键点索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def eye_aspect_ratio(eye):
    # 计算眼睛纵向的两组点的欧氏距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算眼睛横向的欧氏距离
    C = dist.euclidean(eye[0], eye[3])
    # 计算眼睛的纵横比
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # 计算嘴巴纵向的欧氏距离
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    # 计算嘴巴横向的欧氏距离
    C = dist.euclidean(mouth[0], mouth[6])
    # 计算嘴巴的纵横比
    mar = (A + B) / (2.0 * C)
    return mar

def detfatigue(frame):
    # 预处理图像以提高检测率
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)  # 添加直方图均衡化
    
    # 尝试多个尺度进行人脸检测
    faces = []
    scales = [0.5, 1.0, 1.5]  # 多尺度检测
    for scale in scales:
        scaled_frame = cv2.resize(frame_gray, None, fx=scale, fy=scale)
        detected = detector(scaled_frame, 0)
        if detected:
            # 将检测结果转换回原始尺度
            faces = [(rect.left()/scale, rect.top()/scale, 
                     rect.right()/scale, rect.bottom()/scale) for rect in detected]
            break
    
    eyear = 0.0
    mouthar = 0.0
    
    if faces:  # 如果检测到人脸
        # 使用最大的人脸
        face_rect = max(faces, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
        left, top, right, bottom = map(int, face_rect)
        
        # 扩大检测区域
        padding_h = int((bottom - top) * 0.1)  # 垂直方向增加10%
        padding_w = int((right - left) * 0.1)  # 水平方向增加10%
        
        # 确保扩展后的坐标不超出图像范围
        height, width = frame.shape[:2]
        rect = dlib.rectangle(
            max(0, left - padding_w),
            max(0, top - padding_h),
            min(width, right + padding_w),
            min(height, bottom + padding_h)
        )
        
        try:
            # 使用改进的关键点检测
            shape = predictor(frame_gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # 获取眼睛和嘴巴坐标
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            
            # 计算眼睛和嘴巴的纵横比
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            eyear = (leftEAR + rightEAR) / 2.0
            mouthar = mouth_aspect_ratio(mouth)
            
            # 绘制面部特征点和轮廓
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # 绘制眼睛和嘴巴轮廓
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            
            # 在画面上显示当前的EAR和MAR值
            cv2.putText(frame, f"EAR: {eyear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mouthar:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        except Exception as e:
            print(f"关键点检测失败: {str(e)}")
            return frame, 0.0, 0.0
    
    return frame, eyear, mouthar