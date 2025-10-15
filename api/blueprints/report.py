import os
import json
import time
from types import SimpleNamespace
import numpy as np
import cv2
from datetime import datetime
from dotenv import load_dotenv

from flask import Blueprint, request, jsonify, current_app, send_from_directory, session
from werkzeug.utils import secure_filename

from api.extensions import db
from utils.detect_upper_body.detector import UpperBodyDetector
from utils.database.models import VideoSliceImage, RecoveryRecord, RecoveryRecordDetail

load_dotenv()

report_bp = Blueprint('report', __name__)

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

@report_bp.route('/detect_upper_body', methods=['POST'])
def detect_upper_body():
    if 'detection_history' not in session:
        session['detection_history'] = {
            'last_result': True,
            'consecutive_false_count': 0
        }
    
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

            user_history = session['detection_history']
            
            if not current_detection:
                detection_history['consecutive_false_count'] += 1
            else:
                detection_history['consecutive_false_count'] = 0
                detection_history['last_result'] = True
            
            response_value = False if detection_history['consecutive_false_count'] >= 2 else True
            
            detection_history['last_result'] = response_value

            session['detection_history'] = user_history
            session.modified = True 
            
            response_data = {"is_upper_body_in_frame": response_value}
            return jsonify(response_data), 200

        except Exception as e:
            print(f"Error during detection: {e}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500


UPLOAD_FOLDER = os.getenv("SLICE_SAVE_PATH", "uploads/slices")
VIDEO_FOLDER = os.getenv("VIDEO_SAVE_PATH", "uploads/video")

@report_bp.route('/api/reports/upload_sprites', methods=['POST'])
def upload_sprite_sheet():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('files')
    record_id = request.form.get('record_id')
    exercise_id = request.form.get('exercise_id') # 旧称 actionCategory

    if not record_id or not exercise_id:
        return jsonify({'error': 'Fields "record_id" and "exercise_id" are required'}), 400

    if not files or files[0].filename == '':
        return jsonify({'message': 'No selected files to upload.'}), 400

    upload_path = UPLOAD_FOLDER
    os.makedirs(upload_path, exist_ok=True)

    saved_files_info = []
    slice_order_start = db.session.query(db.func.max(VideoSliceImage.slice_order)).filter_by(record_id=record_id, exercise_id=exercise_id).scalar() or 0

    for i, file in enumerate(files, 1):
        filename = secure_filename(f"rec{record_id}_ex{exercise_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.jpg")
        absolute_filepath = os.path.join(upload_path, filename)
        file.save(absolute_filepath)
        
        db_relative_path = os.path.join(UPLOAD_FOLDER, filename)
        
        slice_image = VideoSliceImage(
            record_id=record_id,
            exercise_id=int(exercise_id),
            slice_order=slice_order_start + i,
            image_path=db_relative_path, # 存入相对路径
            timestamp=datetime.now()
        )
        db.session.add(slice_image)
        saved_files_info.append(db_relative_path)
    
    try:
        db.session.commit()
        return jsonify({
            'message': f'Successfully uploaded {len(saved_files_info)} sprite sheets for exercise {exercise_id}.',
            'files_saved': saved_files_info
        }), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Database commit failed in upload_sprite_sheet: {e}")
        return jsonify({'error': f'Failed to save records to database: {str(e)}'}), 500

@report_bp.route('/api/reports/evaluate', methods=['POST'])
def evaluate_exercise_report():
    data = request.json
    record_id = data.get('record_id')
    exercise_id = data.get('exercise_id')

    if not record_id or not exercise_id:
        return jsonify({"error": "Fields 'record_id' and 'exercise_id' are required"}), 400

    try:
        result = current_app.report_service.evaluate_and_save(
            record_id=int(record_id),
            exercise_id=int(exercise_id)
        )
        
        if result.get('success'):
            return jsonify(result), 201
        else:
            error_msg = result.get('error', 'Unknown processing error')
            return jsonify(result), 404 if "未找到" in error_msg else 500

    except Exception as e:
        current_app.logger.error(f"An unexpected error in evaluate_exercise_report: {e}")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

@report_bp.route('/api/reports/<int:record_id>/summarize', methods=['POST'])
def summarize_and_save_report(record_id):
    try:
        result = current_app.report_service.summarize_and_save_for_record(record_id)
        
        if result.get('success'):
            return jsonify(result), 200
        else:
            error_msg = result.get('error', 'Unknown processing error')
            return jsonify(result), 404 if "not found" in error_msg.lower() else 400

    except Exception as e:
        current_app.logger.error(f"An unexpected error in summarize_and_save_report: {e}")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500
