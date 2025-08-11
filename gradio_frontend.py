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
        # 添加状态跟踪变量
        self.fatigue_counter = 0
        self.yawn_counter = 0
        self.fatigue_detected = False
        self.yawn_detected = False
        self.last_fatigue_time = -1
        self.last_yawn_time = -1

    def process_video(self, input_video, progress=gr.Progress()):
        """处理上传的视频文件，进行疲劳检测"""
        if input_video is None:
            return None, "请先上传视频文件", "", ""
        
        # 重置统计数据
        self.detection_results = []
        self.total_frames = 0
        self.fatigue_frames = 0
        self.yawn_frames = 0
        # 重置状态跟踪变量
        self.fatigue_counter = 0
        self.yawn_counter = 0
        self.fatigue_detected = False
        self.yawn_detected = False
        self.last_fatigue_time = -1
        self.last_yawn_time = -1

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

        # 调整检测阈值和参数，提高检测灵敏度
        EAR_THRESHOLD = 0.3  # 眼睛闭合阈值
        MAR_THRESHOLD = 0.5   # 降低打哈欠阈值，提高检测灵敏度
        FATIGUE_CONSEC_FRAMES = 15  # 降低连续帧要求
        YAWN_CONSEC_FRAMES = 10     # 降低连续帧要求
        MIN_INTERVAL_FRAMES = fps * 1  # 降低最小间隔到1秒

        # 连续帧计数器
        fatigue_frame_count = 0
        yawn_frame_count = 0

        # 添加详细的调试信息
        print(f"视频处理开始:")
        print(f"- 视频尺寸: {width}x{height}")
        print(f"- 帧率: {fps} fps")
        print(f"- 总帧数: {total_frames}")
        print(f"- 检测阈值: EAR < {EAR_THRESHOLD}, MAR > {MAR_THRESHOLD}")
        print(f"- 确认帧数: 疲劳{FATIGUE_CONSEC_FRAMES}帧, 打哈欠{YAWN_CONSEC_FRAMES}帧")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 进行疲劳检测
                processed_frame, ear, mar = detfatigue(frame.copy())
                current_time = frame_count / fps

                # 改进的疲劳检测逻辑
                current_fatigue = ear < EAR_THRESHOLD and ear > 0
                current_yawn = mar > MAR_THRESHOLD and mar > 0

                # 疲劳状态检测（需要连续帧确认）
                if current_fatigue:
                    fatigue_frame_count += 1
                else:
                    fatigue_frame_count = 0

                # 打哈欠检测（需要连续帧确认）
                if current_yawn:
                    yawn_frame_count += 1
                else:
                    yawn_frame_count = 0

                # 确认疲劳状态
                is_fatigue_confirmed = fatigue_frame_count >= FATIGUE_CONSEC_FRAMES
                is_yawn_confirmed = yawn_frame_count >= YAWN_CONSEC_FRAMES

                # 添加状态文本
                status_text = ""
                fatigue_logged = False
                yawn_logged = False

                # 疲劳检测逻辑
                if is_fatigue_confirmed:
                    # 检查是否需要记录新的疲劳事件
                    if self.last_fatigue_time == -1 or (frame_count - self.last_fatigue_time) >= MIN_INTERVAL_FRAMES:
                        status_text = "检测到疲劳 - 眼睛闭合"
                        self.fatigue_frames += 1
                        self.last_fatigue_time = frame_count
                        fatigue_logged = True

                    cv2.putText(processed_frame, "FATIGUE DETECTED!", (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # 打哈欠检测逻辑
                if is_yawn_confirmed:
                    # 检查是否需要记录新的打哈欠事件
                    if self.last_yawn_time == -1 or (frame_count - self.last_yawn_time) >= MIN_INTERVAL_FRAMES:
                        status_text += " | 检测到打哈欠" if status_text else "检测到打哈欠"
                        self.yawn_frames += 1
                        self.last_yawn_time = frame_count
                        yawn_logged = True

                    cv2.putText(processed_frame, "YAWNING DETECTED!", (10, 140),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

                if not status_text:
                    status_text = "正常状态"
                    cv2.putText(processed_frame, "NORMAL", (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                # 显示连续帧计数（用于调试）
                if fatigue_frame_count > 0:
                    cv2.putText(processed_frame, f"Fatigue frames: {fatigue_frame_count}/{FATIGUE_CONSEC_FRAMES}",
                              (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if yawn_frame_count > 0:
                    cv2.putText(processed_frame, f"Yawn frames: {yawn_frame_count}/{YAWN_CONSEC_FRAMES}",
                              (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 添加时间戳
                timestamp_text = f"Time: {current_time:.1f}s Frame: {frame_count}/{total_frames}"
                cv2.putText(processed_frame, timestamp_text, (10, height-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 只在确认检测到事件时记录结果
                if fatigue_logged or yawn_logged:
                    self.detection_results.append({
                        'frame': frame_count,
                        'timestamp': current_time,
                        'ear': ear,
                        'mar': mar,
                        'is_fatigue': fatigue_logged,
                        'is_yawn': yawn_logged,
                        'status': status_text
                    })

                # 写入输出视频
                out.write(processed_frame)
                
                frame_count += 1
                self.total_frames = frame_count
                
                # 使用Gradio进度条显示百分比进度
                if frame_count % 10 == 0:  # 每10帧更新一次进度
                    current_time = frame_count / fps
                    total_time = total_frames / fps
                    progress_ratio = (current_time / total_time) if total_time > 0 else 0
                    progress_text = f"处理进度: {progress_ratio*100:.1f}% ({current_time:.1f}s/{total_time:.1f}s)"
                    progress(progress_ratio, desc=progress_text)

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
        if self.total_frames == 0:
            return "暂无检测数据"
        
        report = "### 疲劳检测详细报告\n\n"
        
        # 按时间顺序显示重要事件
        fatigue_events = []
        yawn_events = []
        
        for result in self.detection_results:
            if result['is_fatigue']:
                fatigue_events.append(f"{result['timestamp']:.1f}s: 检测到疲劳 (眼睛比值:: {result['ear']:.3f})")
            if result['is_yawn']:
                yawn_events.append(f"{result['timestamp']:.1f}s: 检测到打哈欠 (嘴巴比值: {result['mar']:.3f})")

        if fatigue_events:
            report += "### 疲劳事件记录\n"
            for event in fatigue_events[:10]:  # 显示前10个事件
                report += f"- {event}\n"
            if len(fatigue_events) > 10:
                report += f"- ... 还有 {len(fatigue_events) - 10} 个疲劳事件\n"
            report += "\n"
        
        if yawn_events:
            report += "### 打哈欠事件记录\n"
            for event in yawn_events[:10]:  # 显示前10个事件
                report += f"- {event}\n"
            if len(yawn_events) > 10:
                report += f"- ... 还有 {len(yawn_events) - 10} 个打哈欠事件\n"
            report += "\n"
        
        if not fatigue_events and not yawn_events:
            report += "### 检测结果\n"
            report += "✅ **一切正常** - 未检测到明显的疲劳或打哈欠行为\n\n"
            report += "### 检测过程\n"
            fps = 30  # 假设30fps
            total_duration = self.total_frames / fps
            report += f"- **处理帧数**: {self.total_frames} 帧\n"
            report += f"- **检测时长**: {total_duration:.1f} 秒\n"
            report += f"- **检测状态**: 全程监控正常\n"
            report += f"- **安全评级**: 驾驶状态良好\n\n"

        return report
    
    def generate_statistics(self):
        """生成统计信息"""
        if self.total_frames == 0:
            return "暂无统计数据"
        
        # 计算实际检测时长
        fps = 30  # 假设30fps
        total_duration = self.total_frames / fps

        # 计算事件统计
        fatigue_events_count = len([r for r in self.detection_results if r['is_fatigue']])
        yawn_events_count = len([r for r in self.detection_results if r['is_yawn']])

        # 计算平均EAR和MAR（从所有有效检测中计算）
        if self.detection_results:
            valid_ears = [r['ear'] for r in self.detection_results if r['ear'] > 0]
            valid_mars = [r['mar'] for r in self.detection_results if r['mar'] > 0]
            avg_ear = sum(valid_ears) / len(valid_ears) if valid_ears else 0
            avg_mar = sum(valid_mars) / len(valid_mars) if valid_mars else 0
        else:
            avg_ear = 0
            avg_mar = 0

        # 基于事件频率评估风险等级
        total_events = fatigue_events_count + yawn_events_count
        events_per_minute = (total_events * 60) / total_duration if total_duration > 0 else 0

        if events_per_minute > 10:
            risk_level = "🔴 **高风险** - 频繁疲劳，建议立即休息"
        elif events_per_minute > 5:
            risk_level = "🟡 **中风险** - 存在疲劳迹象，建议注意休息"
        elif events_per_minute > 0:
            risk_level = "🟢 **低风险** - 偶有疲劳，保持警惕"
        else:
            risk_level = "✅ **无风险** - 状态良好"

        stats = f"""### 检测统计信息

### 基本信息
- **检测时长**: {total_duration:.1f} 秒
- **总帧数**: {self.total_frames}
- **疲劳事件数**: {fatigue_events_count} 次
- **打哈欠事件数**: {yawn_events_count} 次

### 风险评估
- {risk_level}
- **事件频率**: {events_per_minute:.1f} 次/分钟

### 生理指标
- **平均眼睛纵横比 (EAR)**: {avg_ear:.3f}
- **平均嘴巴纵横比 (MAR)**: {avg_mar:.3f}

### 检测阈值
- **疲劳检测阈值 (EAR)**: < 0.25
- **打哈欠检测阈值 (MAR)**: > 0.6
- **连续帧确认**: 疲劳15帧，打哈欠10帧
"""

        return stats

    def clear_all(self):
        """清除所有数据和界面内容"""
        # 重置所有统计数据
        self.detection_results = []
        self.total_frames = 0
        self.fatigue_frames = 0
        self.yawn_frames = 0
        self.fatigue_counter = 0
        self.yawn_counter = 0
        self.fatigue_detected = False
        self.yawn_detected = False
        self.last_fatigue_time = -1
        self.last_yawn_time = -1

        return (
            None,  # input_video
            None,  # output_video
            "等待上传视频...",  # status_text
            "上传视频并开始检测后，这里将显示详细的检测报告",  # report_output
            "上传视频并开始检测后，这里将显示统计信息"  # statistics_output
        )

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
                    
                    with gr.Row():
                        process_btn = gr.Button(
                            " 开始疲劳检测",
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            " 清除",
                            variant="secondary",
                            size="lg"
                        )

                    gr.Markdown("### 处理状态")
                    status_text = gr.Textbox(
                        label="处理状态",
                        value="等待上传视频...",
                        interactive=False
                    )

                    # 添加示例视频选择
                    gr.Markdown("### 示例视频")
                    with gr.Row():
                        example_video_1 = gr.Video(
                            value="video/1.mp4",
                            label="示例视频 1",
                            height=200,
                            interactive=False
                        )
                        example_video_2 = gr.Video(
                            value="video/2.mp4",
                            label="示例视频 2",
                            height=200,
                            interactive=False
                        )
                        example_video_3 = gr.Video(
                            value="video/3.mp4",
                            label="示例视频 3",
                            height=200,
                            interactive=False
                        )

                    with gr.Row():
                        load_example_1_btn = gr.Button("使用示例1", size="sm")
                        load_example_2_btn = gr.Button("使用示例2", size="sm")
                        load_example_3_btn = gr.Button("使用示例3", size="sm")

                # 右侧：检测结果区域
                with gr.Column(scale=2):
                    gr.Markdown("### 检测结果")
                    
                    with gr.Tabs():
                        with gr.Tab("统计信息"):
                            statistics_output = gr.Markdown(
                                value="上传视频并开始检测后，这里将显示统计信息",
                                height=320
                            )
                        
                        with gr.Tab("详细报告"):
                            report_output = gr.Markdown(
                                value="上传视频并开始检测后，这里将显示详细的检测报告",
                                height=320
                            )

                    # 将使用说明和检测指标移到右侧
                    gr.Markdown("### 使用说明")
                    gr.Markdown("""
                    1. 点击左侧"上传视频文件"按钮，选择要检测的视频
                    2. 或者选择下方的示例视频并点击"加载示例视频"
                    3. 点击"开始疲劳检测"按钮开始处理
                    4. 处理完成后，左侧将显示带有检测结果的视频
                    5. 右侧将显示详细的统计信息和检测报告
                    """)

                    gr.Markdown("### 检测指标")
                    gr.Markdown("""
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

            clear_btn.click(
                fn=self.clear_all,
                inputs=[],
                outputs=[input_video, output_video, status_text, report_output, statistics_output]
            )

            # 加载示例视频事件
            load_example_1_btn.click(
                fn=lambda: "video/1.mp4",
                outputs=[input_video]
            )
            load_example_2_btn.click(
                fn=lambda: "video/2.mp4",
                outputs=[input_video]
            )
            load_example_3_btn.click(
                fn=lambda: "video/3.mp4",
                outputs=[input_video]
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
