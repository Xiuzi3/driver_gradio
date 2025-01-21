
## 项目介绍
该项目为疲劳检测系统
疲劳检测部分，使用Dlib进行人脸关键点检测，同时使用Yolo11检测面部特征，然后通过计算眼睛和嘴巴的开合程度来判断是存在否闭眼或者打哈欠，并使用Perclos模型计算疲劳程度。

## 使用方法
依赖：YoloV11、Dlib、PySide2

直接运行main.py，即可使用本程序。

pip install -r requirements.txt 安装所有依赖，python环境为3.10或3.9



