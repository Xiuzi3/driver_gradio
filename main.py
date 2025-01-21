#主函数
import sys
import cv2
import time
import myframe
from PySide2 import QtWidgets
from PySide2.QtWidgets import QMainWindow, QApplication
from PySide2.QtCore import QTimer, QSize
from PySide2.QtGui import QImage, QPixmap, QFont
from PySide2.QtCore import Qt as QtCore
from PySide2.QtCore import Qt  # 添加这行
from ui_mainwindow import Ui_MainWindow

# 定义变量
EYE_AR_THRESH = 0.15        # 眼睛长宽比
EYE_AR_CONSEC_FRAMES = 2    # 闪烁阈值
MAR_THRESH = 0.5            # 打哈欠阈值
MOUTH_AR_CONSEC_FRAMES = 3  # 闪烁阈值
frame_counter = 0
blink_counter = 0           # 眨眼计数器
yawn_counter = 0           # 哈欠计数器

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.f_type = 0
        self.cap = None  # 初始化摄像头对象为None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(16)
        self.last_time = time.time()
        
        # 简化阈值设置
        self.EYE_THRESH = 0.3         # 眨眼阈值
        self.CLOSED_EYE_THRESH = 0.27  # 疲劳判定阈值
        self.MIN_BLINK_FRAMES = 2      # 最少闭眼帧数
        self.MAX_BLINK_FRAMES = 7      # 最大闭眼帧数
        self.CLOSED_EYE_FRAMES = 8     # 持续闭眼判定帧数
        self.EYE_CONFIRM_FRAMES = 2    # 状态确认帧数
        
        # 哈欠检测相关
        self.MOUTH_THRESH = 0.6        # 哈欠阈值
        self.MIN_YAWN_FRAMES = 15      # 最少哈欠帧数
        self.YAWN_INTERVAL = 60        # 哈欠检测时间窗口（秒）
        self.MAX_YAWN_COUNT = 3        # 时间窗口内的哈欠次数阈值
        
        # 计数器重置时间
        self.RESET_INTERVAL = 300      # 5分钟重置一次计数
        self.last_reset_time = time.time()
        
        # 状态变量
        self.eye_closed_confirm = 0
        self.eye_open_confirm = 0
        self.last_eye_state = 'open'
        self.COUNTER = 0
        self.mCOUNTER = 0
        
        # 记录列表
        self.blink_times = []
        self.yawn_times = []
        
        # EAR历史记录
        self.last_ear_values = []
        self.EAR_HISTORY_LENGTH = 5
        self.min_valid_ear = 0.2
        
        # 初始化计数器和状态变量
        self.TOTAL = 0
        self.mTOTAL = 0
        self.start_time = time.time()
        self.blink_rate = 0
        self.yawn_rate = 0
        self.last_blink_check = time.time()
        self.last_yawn_check = time.time()
        self.last_log_time = time.time()
        
        # 初始化眼睛状态确认计数器
        self.eye_closed_confirm = 0
        self.eye_open_confirm = 0
        
        # 以0.27为疲劳判定标准调整阈值
        self.EYE_THRESH = 0.3         # 眨眼阈值
        self.CLOSED_EYE_THRESH = 0.27  # 将0.27设为疲劳判定阈值
        self.MIN_BLINK_FRAMES = 2      # 最少闭眼帧数
        self.MAX_BLINK_FRAMES = 7      # 最大闭眼帧数
        self.CLOSED_EYE_FRAMES = 8     # 持续闭眼判定帧数
        self.EYE_CONFIRM_FRAMES = 2    # 状态确认帧数
        self.FATIGUE_INTERVAL = 20     # 疲劳检测时间窗口
        
        # 疲劳分数相关
        self.fatigue_score = 0.0
        self.fatigue_decay_rate = 0.05
        self.min_valid_ear = 0.2       # 最小有效EAR值
        self.fatigue_threshold = 45     # 疲劳分数阈值
        
        # 更新状态阈值
        self.ear_status = {
            'normal': (0.3, '正常'),       # > 0.3 正常
            'tired': (0.27, '疲劳'),       # <= 0.27 疲劳
            'very_tired': (0.25, '重度疲劳'), # <= 0.25 重度疲劳
            'closed': (0.2, '闭眼')         # <= 0.2 闭眼
        }
        
        # 添加EAR历史记录
        self.last_ear_values = []      # 初始化EAR历史记录列表
        self.EAR_HISTORY_LENGTH = 5    # 设置历史记录长度
        self.consecutive_detections = 0 # 连续检测计数
        
        # 初始化标签
        self.init_labels()
        self.actionOpen_camera.triggered.connect(self.CamConfig_init)
        self.setup_labels_style()

    def init_labels(self):
        self.label.setText("请打开摄像头")
        self.label_2.setText("疲劳检测：")
        self.label_3.setText("眨眼次数：0")
        self.label_4.setText("哈欠次数：0")
        self.label_5.setText("面部特征检测：")
        self.label_6.setText("")
        self.label_7.setText("")
        self.label_8.setText("")
        self.label_9.setText("未检测到面部特征")
        self.label_10.setText("当前状态：清醒")
        
        # 设置标签样式
        self.label_10.setStyleSheet("QLabel { color: green; }")

    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_time < 1/30:  # 限制帧率
            return None
        
        self.last_time = current_time
        
        # 添加处理时间日志
        start_process_time = time.time()
        
        frame = cv2.resize(frame, (640, 480))
        
        # 记录处理耗时
        process_time = time.time() - start_process_time
        if process_time > 0.1:  # 果处理时间超过100ms，记录日志
            self.log_event(f"警告：帧处理耗时长 - {process_time:.3f}秒")
        
        return frame

    def showImage(self, img):
        """显示图像到QLabel"""
        try:
            height, width = img.shape[:2]
            if img.ndim == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
            temp_pixmap = QPixmap.fromImage(temp_image)
            
            # 获取label的大小
            label_size = self.label.size()
            # 保持纵横比缩放图片
            scaled_pixmap = temp_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.log_event(f"显示图像错误: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()
            event.accept()
        except Exception as e:
            self.log_event(f"关闭窗口误: {str(e)}")
            event.accept()

    def update_frame(self):
        if self.f_type == 1 and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    frame = self.process_frame(frame)
                    if frame is None:
                        return
                    
                    # 面部特征检测
                    frame, eye, mouth, labellist = myframe.process(frame)
                    
                    # 如果检测失败，重置计数器
                    if eye is None or mouth is None:
                        self.COUNTER = 0
                        self.last_eye_state = 'open'
                        self.label_9.setText("未能检测到面部特征")
                        # 仍然显示原始帧
                        show = cv2.resize(frame, (640, 480))
                        self.showImage(show)
                        return
                    
                    # 更新状态
                    self.update_fatigue_status(eye, mouth)
                    self.update_behavior_detection(labellist)
                    
                    # 显示帧
                    show = cv2.resize(frame, (640, 480))
                    self.showImage(show)
                    
                except Exception as e:
                    self.log_event(f"处理帧错误: {str(e)}")
                    self.COUNTER = 0
                    self.last_eye_state = 'open'
                    # 发生错误时显示原始帧
                    show = cv2.resize(frame, (640, 480))
                    self.showImage(show)

    def log_event(self, message):
        """添加日志到文本框"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[{current_time}] {message}\n"
        self.textBrowser.append(log_message)

    def update_fatigue_status(self, eye, mouth):
        current_time = time.time()
        
        # 定期重置计数器
        if current_time - self.last_reset_time >= self.RESET_INTERVAL:
            self.blink_times = []
            self.yawn_times = []
            self.last_reset_time = current_time
            self.label_3.setText("眨眼次数：0")
            self.label_4.setText("哈欠次数：0")
            self.log_event("计数器已重置")
        
        # 清理超出时间窗口的记录
        self.blink_times = [t for t in self.blink_times if current_time - t <= self.RESET_INTERVAL]
        self.yawn_times = [t for t in self.yawn_times if current_time - t <= self.YAWN_INTERVAL]
        
        if eye < self.min_valid_ear or eye == 0:
            self.eye_closed_confirm = 0
            self.eye_open_confirm = 0
            self.COUNTER = 0
            self.last_eye_state = 'open'
            self.last_ear_values = []
            self.label_3.setText("眨眼次数：未检测")
            self.label_10.setText("未能检测到面部特征\n请调整面部角度")
            self.label_10.setStyleSheet("QLabel { color: orange; }")
            return
            
        # 更新EAR历史记录和计算平均值
        self.last_ear_values.append(eye)
        if len(self.last_ear_values) > self.EAR_HISTORY_LENGTH:
            self.last_ear_values.pop(0)
            
        valid_ears = [e for e in self.last_ear_values if e >= self.min_valid_ear]
        if not valid_ears:
            return
        avg_ear = sum(valid_ears) / len(valid_ears)
        
        # 眼睛状态检测
        if avg_ear <= self.CLOSED_EYE_THRESH:  # 疲劳状态
            self.eye_closed_confirm += 1
            self.eye_open_confirm = 0
            
            if self.eye_closed_confirm >= self.EYE_CONFIRM_FRAMES:
                if self.last_eye_state == 'open':
                    self.COUNTER = 1
                    self.last_eye_state = 'closed'
                    self.log_event(f"检测到疲劳状态 - EAR: {avg_ear:.3f}")
                else:
                    self.COUNTER += 1
        else:  # EAR > 0.27，清醒状态
            self.eye_open_confirm += 1
            self.eye_closed_confirm = 0
            
            if self.eye_open_confirm >= self.EYE_CONFIRM_FRAMES:
                if self.last_eye_state == 'closed':
                    if self.MIN_BLINK_FRAMES <= self.COUNTER <= self.MAX_BLINK_FRAMES:
                        self.blink_times.append(current_time)
                        self.TOTAL += 1
                        self.label_3.setText(f"眨眼次数：{len(self.blink_times)}")
                self.COUNTER = 0
                self.last_eye_state = 'open'
        
        # 哈欠检测
        if mouth > self.MOUTH_THRESH:
            self.mCOUNTER += 1
        else:
            if self.mCOUNTER >= self.MIN_YAWN_FRAMES:
                self.yawn_times.append(current_time)
                self.label_4.setText(f"哈欠次数：{len(self.yawn_times)}")
                self.log_event(f"检测到哈欠 - MAR: {mouth:.3f}")
            self.mCOUNTER = 0
        
        # 判断疲劳状态
        is_fatigue = False
        fatigue_signs = []
        
        # 基于EAR的疲劳判断
        if avg_ear <= self.CLOSED_EYE_THRESH:
            is_fatigue = True
            fatigue_signs.append(f"EAR值过低({avg_ear:.3f})")
        
        # 基于哈欠的疲劳判断
        recent_yawns = len(self.yawn_times)
        if recent_yawns >= self.MAX_YAWN_COUNT:
            is_fatigue = True
            fatigue_signs.append(f"频繁哈欠({recent_yawns}次/{self.YAWN_INTERVAL}秒)")
        
        # 更新调试信息
        debug_text = f"当前EAR: {eye:.3f}\n"
        debug_text += f"平均EAR: {avg_ear:.3f}\n"
        debug_text += f"当前MAR: {mouth:.3f}\n"
        debug_text += f"哈欠计数: {self.mCOUNTER}\n"
        debug_text += f"状态: {'疲劳' if is_fatigue else '清醒'}"
        self.label_9.setText(debug_text)
        
        # 更新UI显示
        if is_fatigue:
            status_text = "警告：检测到疲劳！\n"
            status_text += "疲劳特征：\n" + "\n".join(fatigue_signs)
            status_text += "\n建议：请注意休息"
            self.label_10.setText(status_text)
            self.label_10.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        else:
            self.label_10.setText(f"当前状态：清醒\nEAR值：{avg_ear:.3f}")
            self.label_10.setStyleSheet("QLabel { color: green; }")

    def update_behavior_detection(self, labellist):
        current_time = time.time()  # 添加当前时间
        
        if labellist:
            # 将标签转换为中文描述
            status_texts = []
            for label in labellist:
                if label == "Closed eye":
                    status_texts.append("眼睛闭合")
                elif label == "Opened eye":
                    status_texts.append("眼睛张开")
                elif label == "Yawn":
                    status_texts.append("打哈欠中")
                elif label == "No-yawn":
                    status_texts.append("嘴巴闭合")
                    
            # 组织显示文本
            if status_texts:
                behavior_text = "当前面部态：\n" + "\n".join(status_texts)
                self.label_9.setText(behavior_text)
                # 根据状态设置颜色
                if "眼睛闭合" in status_texts or "打哈欠中" in status_texts:
                    self.label_9.setStyleSheet("QLabel { color: red; font-size: 12pt; }")
                else:
                    self.label_9.setStyleSheet("QLabel { color: green; font-size: 12pt; }")
                # 记录到日志
                if current_time - self.last_log_time > 3:  # 避免日志刷新太快
                    self.log_event(f"面部状态: {', '.join(status_texts)}")
                    self.last_log_time = current_time
        else:
            self.label_9.setText("未检测到面部特征")
            self.label_9.setStyleSheet("QLabel { color: black; font-size: 12pt; }")

    def CamConfig_init(self):
        if self.f_type == 0:
            self.cap = cv2.VideoCapture(0)  # 打开摄像头
            if not self.cap.isOpened():
                self.log_event("错误：无法打开摄像头")
                return
            self.f_type = 1
            self.timer.start()
            self.log_event("摄像头已打开")
        else:
            self.f_type = 0
            self.timer.stop()
            if self.cap:
                self.cap.release()  # 放摄像头
            self.label.setText("请打开摄像头")
            self.log_event("摄像头已关闭")

    def setup_labels_style(self):
        # 设置标签自动换行和宽度
        self.label_9.setWordWrap(True)
        self.label_9.setMinimumWidth(200)
        self.label_10.setWordWrap(True)
        self.label_10.setMinimumWidth(200)
        
        # 设置标签最大高度
        self.label_9.setMaximumHeight(60)
        self.label_10.setMaximumHeight(60)
        
        # 设置字体
        font = QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_10.setFont(font)
        
        # 使用 QtCore 代替 Qt
        self.label_9.setAlignment(QtCore.AlignLeft | QtCore.AlignVCenter)
        self.label_10.setAlignment(QtCore.AlignLeft | QtCore.AlignVCenter)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())