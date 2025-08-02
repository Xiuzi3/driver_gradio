import gradio as gr
import cv2
import tempfile
from myfatigue import detfatigue
import os
import time

class FatigueDetectionSystem:
    def __init__(self):
        self.detection_results = []
        self.total_frames = 0
        self.fatigue_frames = 0
        self.yawn_frames = 0
        
    def process_video(self, input_video):
        """处理上传的视频文件，进行疲劳检测"""
        if input_video is None:
            return None, "请先上传视频文件", "", ""
        
        # 重置统计数据
        self.detection_results = []
        self.total_frames = 0
        self.fatigue_frames = 0
        self.yawn_frames = 0
        
        # 打开视频文件
        cap = cv2.VideoCapture(input_video)
        
        if not cap.isOpened():
            return None, "无法打开视频文件", "", ""
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # 默认帧率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建输出文件路径，使用H264编码确保兼容性
        timestamp = int(time.time())
        output_filename = f"fatigue_detection_result_{timestamp}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # 使用H264编码器，兼容性更好
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 如果H264不可用，尝试其他编码器
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 如果还是不行，尝试MP4V
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            return None, "无法创建输出视频文件", "", ""

        frame_count = 0

        # 疲劳检测阈值
        EAR_THRESHOLD = 0.25  # 眼睛纵横比阈值
        MAR_THRESHOLD = 0.6   # 嘴巴纵横比阈值（打哈欠）
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 进行疲劳检测
                processed_frame, ear, mar = detfatigue(frame.copy())
                
                # 判断疲劳状态
                is_fatigue = ear < EAR_THRESHOLD and ear > 0
                is_yawn = mar > MAR_THRESHOLD
                
                # 添加状态文本
                status_text = ""
                if is_fatigue:
                    status_text = "检测到疲劳 - 眼睛闭合"
                    cv2.putText(processed_frame, "FATIGUE DETECTED!", (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    self.fatigue_frames += 1
                    
                if is_yawn:
                    status_text += " | 检测到打哈欠" if status_text else "检测到打哈欠"
                    cv2.putText(processed_frame, "YAWNING DETECTED!", (10, 140),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                    self.yawn_frames += 1
                
                if not status_text:
                    status_text = "正常状态"
                    cv2.putText(processed_frame, "NORMAL", (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                # 添加时间戳
                timestamp = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(processed_frame, timestamp, (10, height-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 记录检测结果
                self.detection_results.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'ear': ear,
                    'mar': mar,
                    'is_fatigue': is_fatigue,
                    'is_yawn': is_yawn,
                    'status': status_text
                })
                
                # 写入输出视频
                out.write(processed_frame)
                
                frame_count += 1
                self.total_frames = frame_count
                
                # 更新进度
                if frame_count % 30 == 0:  # 每30帧更新一次进度
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    progress_info = f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})"
                    print(progress_info)
        
        except Exception as e:
            cap.release()
            out.release()
            if os.path.exists(output_path):
                os.remove(output_path)
            return None, f"处理视频时出错: {str(e)}", "", ""
        
        finally:
            cap.release()
            out.release()
        
        # 检查输出文件是否生成成功
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return None, "输出视频文件生成失败", "", ""

        # 生成检测报告
        detection_report = self.generate_detection_report()
        statistics = self.generate_statistics()
        
        return output_path, f"视频处理完成！输出文件: {output_filename}", detection_report, statistics

    def generate_detection_report(self):
        """生成详细的检测报告"""
        if not self.detection_results:
            return "暂无检测数据"
        
        report = "### 疲劳检测详细报告\n\n"
        
        # 按时间顺序显示重要事件
        fatigue_events = []
        yawn_events = []
        
        for result in self.detection_results:
            if result['is_fatigue']:
                fatigue_events.append(f" {result['timestamp']:.1f}s: 检测到疲劳 (EAR: {result['ear']:.3f})")
            if result['is_yawn']:
                yawn_events.append(f" {result['timestamp']:.1f}s: 检测到打哈欠 (MAR: {result['mar']:.3f})")
        
        if fatigue_events:
            report += ""
            for event in fatigue_events[:10]:  # 显示前10个事件
                report += f"- {event}\n"
            if len(fatigue_events) > 10:
                report += f"- ... 还有 {len(fatigue_events) - 10} 个疲劳事件\n"
            report += "\n"
        
        if yawn_events:
            report += "###  打哈欠事件记录\n"
            for event in yawn_events[:10]:  # 显示前10个事件
                report += f"- {event}\n"
            if len(yawn_events) > 10:
                report += f"- ... 还有 {len(yawn_events) - 10} 个打哈欠事件\n"
            report += "\n"
        
        if not fatigue_events and not yawn_events:
            report += "###  检测结果\n未检测到明显的疲劳或打哈欠行为\n\n"
        
        return report
    
    def generate_statistics(self):
        """生成统计信息"""
        if not self.detection_results:
            return "暂无统计数据"
        
        total_duration = len(self.detection_results) / 30  # 假设30fps
        fatigue_percentage = (self.fatigue_frames / self.total_frames) * 100
        yawn_percentage = (self.yawn_frames / self.total_frames) * 100
        
        # 计算平均EAR和MAR
        total_ear = sum(r['ear'] for r in self.detection_results if r['ear'] > 0)
        valid_ear_count = sum(1 for r in self.detection_results if r['ear'] > 0)
        avg_ear = total_ear / valid_ear_count if valid_ear_count > 0 else 0
        
        total_mar = sum(r['mar'] for r in self.detection_results if r['mar'] > 0)
        valid_mar_count = sum(1 for r in self.detection_results if r['mar'] > 0)
        avg_mar = total_mar / valid_mar_count if valid_mar_count > 0 else 0
        
        stats = f""" 

### 基本信息
- **检测时长**: {total_duration:.1f} 秒
- **总帧数**: {self.total_frames}
- **疲劳帧数**: {self.fatigue_frames} ({fatigue_percentage:.2f}%)
- **打哈欠帧数**: {self.yawn_frames} ({yawn_percentage:.2f}%)

### 生理指标
- **平均眼睛纵横比 (EAR)**: {avg_ear:.3f}
- **平均嘴巴纵横比 (MAR)**: {avg_mar:.3f}

### 风险评估
"""
        
        # 风险等级评估
        if fatigue_percentage > 30:
            risk_level = "🔴 **高风险** - 频繁疲劳，建议立即休息"
        elif fatigue_percentage > 10:
            risk_level = "🟡 **中风险** - 存在疲劳迹象，建议注意休息"
        elif fatigue_percentage > 0:
            risk_level = "🟢 **低风险** - 偶有疲劳，保持警惕"
        else:
            risk_level = "✅ **无风险** - 状态良好"
        
        stats += f"- {risk_level}\n"
        
        return stats

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="疲劳检测系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            <div align="center">
                <h1>驾驶员疲劳检测系统</h1>
                <p>上传视频文件进行疲劳检测，系统将自动分析眼睛闭合和打哈欠行为</p>
            </div>
            """)
            
            with gr.Row():
                # 左侧：视频处理区域
                with gr.Column(scale=3):
                    gr.Markdown("### 视频处理")
                    
                    with gr.Row():
                        input_video = gr.Video(
                            label="上传视频文件",
                            height=300
                        )
                        output_video = gr.Video(
                            label="检测结果视频",
                            height=300
                        )
                    
                    process_btn = gr.Button(
                        " 开始疲劳检测",
                        variant="primary",
                        size="lg"
                    )
                    
                    status_text = gr.Textbox(
                        label="处理状态",
                        value="等待上传视频...",
                        interactive=False
                    )
                
                # 右侧：检测结果区域
                with gr.Column(scale=2):
                    gr.Markdown("### 检测结果")
                    
                    with gr.Tabs():
                        with gr.Tab("统计信息"):
                            statistics_output = gr.Markdown(
                                value="上传视频并开始检测后，这里将显示统计信息",
                                height=400
                            )
                        
                        with gr.Tab("详细报告"):
                            report_output = gr.Markdown(
                                value="上传视频并开始检测后，这里将显示详细的检测报告",
                                height=400
                            )
            
            # 说明信息
            gr.Markdown("""
            ### 使用说明
            1. 点击左侧"上传视频文件"按钮，选择要检测的视频
            2. 点击"开始疲劳检测"按钮开始处理
            3. 处理完成后，左侧将显示带有检测结果的视频
            4. 右侧将显示详细的统计信息和检测报告
            
            ### 检测指标
            - **EAR (眼睛纵横比)**: < 0.25 表示眼睛闭合/疲劳
            - **MAR (嘴巴纵横比)**: > 0.6 表示打哈欠
            - **风险等级**: 根据疲劳帧占比自动评估
            """)
            
            # 绑定事件
            process_btn.click(
                fn=self.process_video,
                inputs=[input_video],
                outputs=[output_video, status_text, report_output, statistics_output]
            )
        
        return demo

def main():
    """主函数"""
    system = FatigueDetectionSystem()
    demo = system.create_interface()
    
    # 启动界面
    demo.launch(
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
