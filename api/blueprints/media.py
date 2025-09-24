import os
import json
import time
from types import SimpleNamespace
import numpy as np
import cv2
from datetime import datetime

from flask import Blueprint, request, jsonify, current_app, send_from_directory, session
from werkzeug.utils import secure_filename

from api.extensions import db
from utils.detect_upper_body.detector import UpperBodyDetector
from utils.database.models import VideoSliceImage, RecoveryRecord, RecoveryRecordDetail

media_bp = Blueprint('media', __name__)

# --- Upper Body Detector Initialization ---
mmpose_config_path="utils/pose_estimation/mmpose_config.json"
mmpose_config = SimpleNamespace(**json.load(open(mmpose_config_path,'r')))
detector_device = "cuda:0"
try:
    detector = UpperBodyDetector(mmpose_config.pose2d_config, mmpose_config.pose2d_checkpoint, device=detector_device, margin_ratio=0.05)
except Exception as e:
    print(f"Failed to initialize UpperBodyDetector: {e}")
    detector = None

detection_history = {
    'last_result': True,
    'consecutive_false_count': 0
}

@media_bp.route('/detect_upper_body', methods=['POST'])
def detect_upper_body():
    global detection_history
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided. Please upload an image with key 'image'."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "Could not decode image. Please ensure it's a valid image format."}), 400

            current_detection = detector.detect(img)
            
            if not current_detection:
                detection_history['consecutive_false_count'] += 1
            else:
                detection_history['consecutive_false_count'] = 0
                detection_history['last_result'] = True
            
            response_value = False if detection_history['consecutive_false_count'] >= 2 else True
            
            detection_history['last_result'] = response_value
            
            response_data = {"is_upper_body_in_frame": response_value}
            return jsonify(response_data), 200

        except Exception as e:
            print(f"Error during detection: {e}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500


UPLOAD_FOLDER = 'uploads'
VIDEO_FOLDER = 'uploads/video'

# @media_bp.before_app_first_request
# def create_upload_folders():
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     if not os.path.exists(VIDEO_FOLDER):
#         os.makedirs(VIDEO_FOLDER)

@media_bp.route('/upload_sprite_sheet', methods=['POST'])
def upload_sprite_sheet():
    # ... (Copy the entire upload_sprite_sheet function here)
    if not request.files:
        return jsonify({'error': 'No files provided in the request'}), 400

    action_category = request.form.get('actionCategory', 'unknown')
    record_id = request.form.get('record_id')
    saved_files = []

    upload_folder = os.path.join(current_app.root_path, '..', UPLOAD_FOLDER)

    for key, file in request.files.items():
        if file.filename == '':
            continue
        if file:
            timestamp = int(time.time() * 1000)
            extension = os.path.splitext(file.filename)[1] if os.path.splitext(file.filename)[1] else '.jpg'
            filename = secure_filename(f"{action_category}_{timestamp}_{key}{extension}")
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # Use relative path for database
            db_filepath = os.path.join(UPLOAD_FOLDER, filename)
            saved_files.append(db_filepath)
            
            slice_image = VideoSliceImage(
                record_id=record_id,
                exercise_id=int(action_category),
                slice_order=len(saved_files),
                image_path=db_filepath,
                timestamp=datetime.now()
            )
            db.session.add(slice_image)
    
    try:
        db.session.commit()
        if saved_files:
            return jsonify({
                'message': f'Successfully uploaded {len(saved_files)} frames for action category {action_category}.',
                'files_saved': saved_files
            }), 200
        else:
            return jsonify({'message': 'No valid files uploaded.'}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
        

def _extract_frames_from_sprite(image_path):
    try:
        # Construct absolute path to read the file
        abs_path = os.path.join(current_app.root_path, '..', image_path)
        sprite_image = cv2.imread(abs_path)
        if sprite_image is None:
            return []
        
        img_height, img_width, _ = sprite_image.shape
        frame_height = img_height // 2
        frame_width = img_width // 3
        
        frames = []
        for i in range(2):
            for j in range(3):
                frame = sprite_image[i*frame_height:(i+1)*frame_height, j*frame_width:(j+1)*frame_width]
                frames.append(frame)
        return frames
    except Exception as e:
        print(f"Error extracting frames from {image_path}: {e}")
        return []

@media_bp.route('/api/process_exercise_video', methods=['POST'])
def process_exercise_video():
    """
    根据 record_id 和 exercise_id 拼接雪碧图为完整视频，
    并创建/更新 recovery_record_details 记录。
    """
    data = request.json
    record_id = data.get('record_id')
    exercise_id = data.get('exercise_id')

    if not record_id or not exercise_id:
        return jsonify({"error": "record_id 和 exercise_id 是必需的"}), 400

    try:
        # 1. 创建一个新的 recovery_record_details 条目
        new_detail = RecoveryRecordDetail(
            record_id=record_id,
            exercise_id=exercise_id,
            completion_timestamp=datetime.now()
        )
        db.session.add(new_detail)
        db.session.commit()

        # --- 修改开始 ---
        # 2. 从 video_slice_images 查找所有相关雪碧图，并按 时间戳 和 顺序号 双重排序
        image_slices = VideoSliceImage.query.filter_by(
            record_id=record_id,
            exercise_id=exercise_id
        ).order_by(VideoSliceImage.timestamp, VideoSliceImage.slice_order).all()
        # --- 修改结束 ---

        if not image_slices:
            new_detail.evaluation_details = "错误：未找到对应的视频切片图像。"
            db.session.commit()
            return jsonify({"error": "未找到与此记录相关的视频切片图像"}), 404

        # 3. 提取所有视频帧
        all_frames = []
        for image_slice in image_slices:
            frames = _extract_frames_from_sprite(image_slice.image_path)
            all_frames.extend(frames)
        
        if not all_frames:
            new_detail.evaluation_details = "错误：从雪碧图提取帧失败。"
            db.session.commit()
            return jsonify({"error": "无法从雪碧图中提取视频帧"}), 500

        # 4. 将所有帧拼接成一个视频 (此部分逻辑不变)
        frame_height, frame_width, _ = all_frames[0].shape
        video_filename = f"record_{record_id}_exercise_{exercise_id}_{int(time.time())}.mp4"
        video_filepath = os.path.join(VIDEO_FOLDER, video_filename)
        
        fps = 6.0
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, (frame_width, frame_height))
        
        for frame in all_frames:
            video_writer.write(frame)
        video_writer.release()

        # 5. 更新 recovery_record_details 条目，保存视频路径 (此部分逻辑不变)
        new_detail.video_path = video_filepath
        db.session.commit()

        return jsonify({
            "message": "视频处理成功并已保存",
            "record_detail_id": new_detail.record_detail_id,
            "video_path": video_filepath
        }), 201

    except Exception as e:
        db.session.rollback()
        print(f"Error processing video: {e}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

@media_bp.route('/uploads/video/<path:filename>')
def serve_video(filename):
    return send_from_directory(os.path.join(current_app.root_path, '..', VIDEO_FOLDER), filename)