# 建议将此类保存在 services/report_generator.py

import os
import cv2
import time
import json
from datetime import datetime, timedelta
import base64
import numpy as np  # 确保导入 numpy
import imageio      # 导入 imageio

from openai import OpenAI
import dashscope
from dashscope import MultiModalConversation
from typing import List, Dict, Any, Optional
from sqlalchemy import or_
from utils.database.models import VideoSliceImage, RecoveryRecordDetail, RecoveryRecord

class ReportGenerator:
    def __init__(self, db_session, max_file_size_mb: int = 10):
        if not db_session: raise ValueError("必须提供一个有效的 db_session。")
        self.db_session = db_session
        
        self.api_key = os.getenv("QWEN_API_KEY")
        self.vl_model_name = os.getenv("QWEN_VL_MODEL_NAME", "qwen-vl-plus")
        
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.text_model_name = os.getenv("QWEN_MODEL_NAME", "qwen-plus")

        # 路径和参数配置
        self.video_save_path = os.getenv("VIDEO_SAVE_PATH", "uploads/videos")
        self.exercise_desc = json.load(open(os.getenv("EXERCISE_DESC_PATH", "knowledge_base/exercise_desc.json"), 'r'))
        self.fps_for_video_eval = int(os.getenv("VIDEO_FPS", 5))
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        os.makedirs(self.video_save_path, exist_ok=True)

    def generate_slice_video_feedback(self, record_id: int, exercise_id: int, start_time: datetime, end_time: datetime) -> str:
        # ... (数据库查询逻辑不变) ...
        image_slices = self.db_session.query(VideoSliceImage).filter(
            VideoSliceImage.record_id == record_id,
            VideoSliceImage.exercise_id == exercise_id,
            VideoSliceImage.timestamp.between(start_time, end_time)
        ).filter(
            or_(
                VideoSliceImage.is_part_of_action == True,
                VideoSliceImage.is_part_of_action.is_(None)
            )
        ).order_by(VideoSliceImage.timestamp).all()

        if not image_slices:
            return "在指定时间段内未找到可分析的动作图片。"

        all_frames = [frame for s in image_slices if os.path.exists(s.image_path) for frame in self._extract_frames_from_sprite(s.image_path)]
        if not all_frames:
            return "无法从有效的图片路径中提取任何帧。"

        temp_video_filename = f"temp_feedback_{record_id}_{exercise_id}_{int(time.time())}.mp4"
        temp_video_filepath = os.path.join(self.video_save_path, temp_video_filename)
        
        # ------------------------------------------------------------------
        # 变更: 修正 generate_slice_video_feedback 的奇数尺寸问题
        # ------------------------------------------------------------------
        try:
            # 1. 获取原始的 (可能是奇数的) 尺寸
            original_height, original_width, _ = all_frames[0].shape # e.g., 313, 207

            # 2. 计算新的、保证为偶数的尺寸
            new_width = original_width
            new_height = original_height
            if new_width % 2 != 0: new_width -= 1   # 207 -> 206
            if new_height % 2 != 0: new_height -= 1 # 313 -> 312
            
            print(f"[Debug] 正在创建临时反馈视频 (imageio): {temp_video_filepath}")
            print(f"[Debug] 临时视频原始尺寸 (H,W): {original_height},{original_width} -> 调整后 (H,W): {new_height},{new_width}")

            writer = imageio.get_writer(
                temp_video_filepath,
                fps=self.fps_for_video_eval,
                codec='libx264',
                pixelformat='yuv420p',
                macro_block_size=None
            )
            
            for frame in all_frames:
                # 确保帧是 uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # 3. 调整帧到新的偶数尺寸
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # 4. 关键: BGR -> RGB
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                writer.append_data(frame_rgb)
                
            writer.close()
            print("[Debug] 临时反馈视频创建成功。")
            
        except Exception as e:
            print(f"[Error] 使用 imageio 创建临时反馈视频失败: {e}")
            if os.path.exists(temp_video_filepath):
                os.remove(temp_video_filepath)
            return "创建分析视频失败。"

        exercise_description = self.exercise_desc[exercise_id]

        system_prompt = (
            "你是一名专业的康复治疗师，请根据上传的短视频片段和标准动作描述，面向正在锻炼的乳腺癌患者给出反馈。你的反馈应该是鼓励性的，并指出一个可以立即改进的关键点。语气轻松并且口语化。反馈需简短，限制在20字以内。"
        )
        user_prompt = f"**标准动作描述**：{exercise_description['name']}：{exercise_description['desc']}\n**患者当前动作片段**："

        try:
            with open(temp_video_filepath, "rb") as video_file:
                base64_video = base64.b64encode(video_file.read()).decode("utf-8")
            video_data_uri = f"data:video/mp4;base64,{base64_video}"
        except Exception as e:
            print(f"读取临时视频失败: {e}")
            return "读取分析视频失败。"
        finally:
            if os.path.exists(temp_video_filepath):
                os.remove(temp_video_filepath)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                'role': 'user',
                'content': [
                    {'video': video_data_uri},
                    {'text': user_prompt}
                ]
            }
        ]
        
        try:
            response = MultiModalConversation.call(model=self.vl_model_name, api_key=self.api_key, messages=messages)
            if response.status_code == 200:
                content_list = response.output.choices[0].message.get("content", [])
                feedback = ""
                for part in content_list:
                    if isinstance(part, dict) and 'text' in part:
                        feedback = part['text']
                        break
                return feedback or "未能生成有效的反馈。"
            else:
                return f"API调用失败: Code={response.status_code}, Message={response.message}"
        except Exception as e:
            return f"调用模型时发生未知错误: {e}"

    def _extract_frames_from_sprite(self, image_path: str) -> List[Any]:
        try:
            # cv2.imread 默认以 BGR 格式读取
            sprite_image = cv2.imread(image_path) 
            if sprite_image is None: return []
            img_height, img_width, _ = sprite_image.shape
            frame_height, frame_width = img_height // 2, img_width // 3
            # 注意: cv2.resize 和 shape 使用 (H, W) 顺序
            # 但 sprite_image[...] 切片是 [y:y+H, x:x+W]
            # frame_width = 207, frame_height = 313
            frames = [sprite_image[i*frame_height:(i+1)*frame_height, j*frame_width:(j+1)*frame_width] 
                      for i in range(2) for j in range(3)]
            # 过滤掉空的切片 (如果计算有误)
            return [f for f in frames if f.shape[0] > 0 and f.shape[1] > 0]
        except Exception as e:
            print(f"[Error] 从 {image_path} 提取帧时发生异常: {e}")
            return []

    # (这个函数在上一版中已经正确，无需修改)
    def _create_video_from_frames(self, frames: List[Any], record_id: int, exercise_id: int) -> Optional[str]:
        if not frames:
            print("[Error] 没有帧可以用来创建视频。")
            return None
        
        try:
            test_frame = frames[0]
            if not isinstance(test_frame, np.ndarray):
                print(f"[Error] 帧不是 numpy 数组, 而是 {type(test_frame)}")
                return None
            original_height, original_width, _ = test_frame.shape # e.g., 313, 207
        except Exception as e:
            print(f"[Error] 检查第一帧时出错: {e}")
            return None

        compression_levels = [0.75, 0.5, 0.25]
        base_filename = f"record_{record_id}_exercise_{exercise_id}_{int(time.time())}"

        for scale in compression_levels:
            # 这个函数已经有正确的 "确保偶数" 逻辑
            new_width = int(original_width * scale)  # e.g., int(207 * 0.75) = 155
            new_height = int(original_height * scale) # e.g., int(313 * 0.75) = 234
            if new_width % 2 != 0: new_width -= 1   # 155 -> 154
            if new_height % 2 != 0: new_height -= 1 # 234 -> 234
            if new_width < 2 or new_height < 2: continue
            
            video_filename = f"{base_filename}_scale_{int(scale*100)}p.mp4"
            video_filepath = os.path.join(self.video_save_path, video_filename)

            try:
                print(f"[Debug] 尝试使用 imageio 写入: {video_filepath} @ {self.fps_for_video_eval} fps, scale: {scale}")
                print(f"[Debug] 压缩视频尺寸 (H,W): {new_height},{new_width}")

                writer = imageio.get_writer(
                    video_filepath,
                    fps=self.fps_for_video_eval,
                    codec='libx264',
                    pixelformat='yuv420p',
                    macro_block_size=None
                )

                for frame in frames:
                    if frame.dtype != np.uint8:
                         frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

                    # 调整到新的、偶数的、压缩的尺寸
                    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # BGR -> RGB
                    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    
                    writer.append_data(frame_rgb)

                writer.close()
                print(f"[Debug] imageio 写入完成: {video_filepath}")

                if os.path.exists(video_filepath) and os.path.getsize(video_filepath) > 0:
                    if os.path.getsize(video_filepath) < self.max_file_size_bytes:
                        print(f"[Success] 成功创建视频 (使用 imageio): {video_filepath}")
                        return video_filepath
                    else:
                        os.remove(video_filepath)
                else:
                    print(f"[Warning] 视频文件创建失败或为空: {video_filepath}")
                    if os.path.exists(video_filepath):
                        os.remove(video_filepath)

            except Exception as e:
                print(f"[Error] 使用 imageio 写入失败 (scale {scale}): {e}")
                if os.path.exists(video_filepath):
                     os.remove(video_filepath)
                continue

        print(f"[Error] 所有压缩尝试均失败 (包括 imageio)。\nrecord_id: {record_id}, exercise_id: {exercise_id}")
        return None

    def _build_dashscope_messages(self, video_filepath: str, exercise_id: int) -> List[Dict[str, Any]]:
        try:
            print(f"[Info] 正在读取并编码视频文件: {video_filepath}")
            with open(video_filepath, "rb") as video_file:
                base64_video = base64.b64encode(video_file.read()).decode("utf-8")
            video_data_uri = f"data:video/mp4;base64,{base64_video}"
        except FileNotFoundError:
            print(f"[Error] 视频文件未找到: {video_filepath}")
            raise

        exercise_description = self.exercise_desc[exercise_id] # 确保 exercise_id 是字符串
        
        system_prompt = (
            "你是一名专业的乳腺癌术后上肢物理康复治疗师。请根据下面提供的康复训练视频和标准动作描述，对患者的动作进行全面评估，并以JSON格式返回一个包含10分制分数（score）和评估报告（report）的对象。"
            "评估要求：1. 动作规范性；2. 动作幅度和节奏；3. 潜在错误；4. 改进建议。"
            "分数（score）应为0到10之间的整数，综合反映动作的完成质量。"
            "报告（report）为不超过150字的综合性评估文本。"
            "输出为json，格式为{'score': '', report: ''}"
        )
        
        user_prompt = f"**标准动作描述**：{exercise_description['name']}：{exercise_description['desc']}\n**康复训练视频**："

        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                'role': 'user', 
                'content': [
                    {'video': video_data_uri},
                    {'text': user_prompt}
                ]
            }
        ]
        
    def evaluate_and_save(self, record_id: int, exercise_id: int) -> Dict[str, Any]:
        print(f"[Info] 开始处理 Record ID: {record_id}, Exercise ID: {exercise_id}")

        video_filepath = None
        
        existing_detail = self.db_session.query(RecoveryRecordDetail).filter_by(
            record_id=record_id, 
            exercise_id=exercise_id
        ).first()

        if existing_detail and existing_detail.video_path:
            print(f"  - 发现已存在的记录 Detail ID: {existing_detail.record_detail_id}，视频路径: {existing_detail.video_path}")
            if os.path.exists(existing_detail.video_path):
                print(f"  - 视频文件存在。将直接使用此文件进行评估。")
                video_filepath = existing_detail.video_path
            else:
                print(f"  - 警告: 数据库中记录的视频文件不存在，将重新生成。")
        
        if video_filepath is None:
            image_slices = self.db_session.query(VideoSliceImage).filter_by(
                record_id=record_id,
                exercise_id=exercise_id
            ).filter(
                or_(
                    VideoSliceImage.is_part_of_action == True,
                    VideoSliceImage.is_part_of_action.is_(None)
                )
            ).order_by(VideoSliceImage.timestamp).all()

            if not image_slices:
                return {'success': False, 'error': f"未找到与 Record ID {record_id} 和 Exercise ID {exercise_id} 相关的图片记录。"}
            
            all_frames = [frame for s in image_slices if os.path.exists(s.image_path) for frame in self._extract_frames_from_sprite(s.image_path)]
            if not all_frames:
                return {'success': False, 'error': "无法从有效的图片路径中提取任何帧。"}

            video_filepath = self._create_video_from_frames(all_frames, record_id, exercise_id)
            if not video_filepath:
                return {'success': False, 'error': "从帧创建视频失败，或压缩后文件大小仍然超标。"}
        
        try:
            print(f"[Info] 视频已最终保存至: '{video_filepath}'。")
            messages = self._build_dashscope_messages(video_filepath, exercise_id)
            response = MultiModalConversation.call(model=self.vl_model_name, api_key=self.api_key, messages=messages)

            if response.status_code == 200:
                evaluation_result = ""
                content_list = response.output.choices[0].message.get("content", [])
                
                for part in content_list:
                    if isinstance(part, dict) and 'text' in part:
                        evaluation_result = part['text']
                        break 
                
                if not evaluation_result:
                    error_msg = f"API调用成功，但返回内容中未找到有效的评估文本。返回内容: {content_list}"
                    return {'success': False, 'error': error_msg, 'video_path': video_filepath}
            else:
                error_msg = f"API调用失败: Code={response.status_code}, Message={response.message}"
                return {'success': False, 'error': error_msg, 'video_path': video_filepath}

            if existing_detail:
                print(f"  - 更新已存在的 Detail ID: {existing_detail.record_detail_id}")
                existing_detail.completion_timestamp = datetime.now()
                existing_detail.video_path = video_filepath
                existing_detail.evaluation_details = evaluation_result
                self.db_session.commit()
                detail_id_to_return = existing_detail.record_detail_id
            else:
                print("  - 创建新的 Detail 记录")
                new_detail = RecoveryRecordDetail(
                    record_id=record_id, exercise_id=exercise_id, completion_timestamp=datetime.now(),
                    video_path=video_filepath, evaluation_details=evaluation_result)
                self.db_session.add(new_detail)
                self.db_session.commit()
                detail_id_to_return = new_detail.record_detail_id
            
            return {'success': True, 'detail_id': detail_id_to_return, 'video_path': video_filepath, 'evaluation': evaluation_result}
        except Exception as e:
            self.db_session.rollback()
            error_details = f"在评估或数据库操作过程中发生未知错误: {type(e).__name__}: {e}"
            return {'success': False, 'error': error_details, 'video_path': video_filepath}
    
    def summarize_rehab_reports(self, reports: List[str]) -> str:
        if not reports:
            return "报告列表为空，无法总结。"

        report_texts = []
        for r in reports:
            try:
                clean_r = r.strip().lstrip("```json").rstrip("```")
                report_json = json.loads(clean_r)
                if 'report' in report_json:
                    report_texts.append(report_json['report'])
            except (json.JSONDecodeError, TypeError):
                if isinstance(r, str) and r.strip().startswith('{'):
                     print(f"[Warning] 无法解析 JSON 报告: {r}")
                elif isinstance(r, str):
                    report_texts.append(r)
        
        if not report_texts:
            return "无法从提供的报告记录中提取有效的文本内容进行总结。"

        combined_reports = "\n---\n".join(report_texts)
        prompt = f"你是一名专业的康复治疗师助理，你的任务是分析和总结多份康复锻炼报告。\n请仔细阅读以下提供的多份关于上肢康复锻炼的独立报告，并将它们的核心信息整合成一份全面、连贯、有条理的综合性总结报告。报告注意分行。字数不超过250字。\n--- 原始报告如下 ---\n{combined_reports}"
        
        messages = [{"role": "user", "content": prompt}]
        
        completion = self.client.chat.completions.create(
            model=self.text_model_name,
            messages=messages
        )
        response = completion.model_dump()
        return response['choices'][0]['message']['content']
    
    def summarize_and_save_for_record(self, record_id: int) -> Dict[str, Any]:
        record = self.db_session.query(RecoveryRecord).filter_by(record_id=record_id).first()
        if not record:
            return {'success': False, 'error': f'Record with ID {record_id} not found.'}

        details = self.db_session.query(RecoveryRecordDetail).filter_by(record_id=record_id).all()
        if not details:
            return {'success': False, 'error': f'No evaluation details found for record ID {record_id} to summarize.'}
        
        reports_to_summarize = [d.evaluation_details for d in details if d.evaluation_details]
        
        if not reports_to_summarize:
            return {'success': False, 'error': 'Found evaluation records, but they contain no text to summarize.'}

        summary_text = self.summarize_rehab_reports(reports_to_summarize)
        
        try:
            record.evaluation_summary = summary_text
            self.db_session.commit()
            print(f"[Success] 已为 Record ID {record_id} 生成并保存了总结报告。")
            return {'success': True, 'record_id': record_id, 'summary': summary_text}
        except Exception as e:
            self.db_session.rollback()
            print(f"[Error] 保存总结报告至 Record ID {record_id} 时发生数据库错误: {e}")
            return {'success': False, 'error': f'Failed to save summary to database: {str(e)}'}