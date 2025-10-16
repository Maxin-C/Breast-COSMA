import os
import json
import time
from types import SimpleNamespace
import numpy as np
import cv2
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image

from flask import Blueprint, request, jsonify, current_app, send_from_directory, session
from werkzeug.utils import secure_filename

from api.extensions import db
from utils.database.models import VideoSliceImage, RecoveryRecord, RecoveryRecordDetail

load_dotenv()

report_bp = Blueprint('report', __name__)

UPLOAD_FOLDER = os.getenv("SLICE_SAVE_PATH", "uploads/slices")
VIDEO_FOLDER = os.getenv("VIDEO_SAVE_PATH", "uploads/videos")

@report_bp.route('/detect_upper_body', methods=['POST'])
def detect_upper_body():
    if 'detection_history' not in session:
        session['detection_history'] = {'consecutive_false_count': 0}

    detector = current_app.pose_inferencer

    if detector is None:
        return jsonify({"error": "Pose detection model is not available."}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    file = request.files['image']
    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400

    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result_generator = detector(img, show=False)
        result = next(result_generator)
        
        current_detection = bool(result['predictions'] and result['predictions'][0])
        
        user_history = session['detection_history']
        if not current_detection:
            user_history['consecutive_false_count'] += 1
        else:
            user_history['consecutive_false_count'] = 0
        
        response_value = False if user_history['consecutive_false_count'] >= 2 else True
        session['detection_history'] = user_history
        session.modified = True
        
        return jsonify({"is_upper_body_in_frame": response_value}), 200

    except Exception as e:
        current_app.logger.error(f"Error during detection: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@report_bp.route('/reports/upload_sprites', methods=['POST'])
def upload_sprite_sheet():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('files')
    record_id = request.form.get('record_id')
    exercise_id = request.form.get('exercise_id')

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

        pil_image = Image.open(absolute_filepath)
        frame_results = current_app.action_classifier_service.predict_frames_in_sprite(
            pil_image,
            int(exercise_id)
        )
        
        slice_image = VideoSliceImage(
            record_id=record_id,
            exercise_id=int(exercise_id),
            slice_order=slice_order_start + i,
            image_path=db_relative_path,
            timestamp=datetime.now(),
            is_part_of_action=(frame_results == 1)
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

@report_bp.route('/reports/evaluate', methods=['POST'])
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

@report_bp.route('/reports/<int:record_id>/summarize', methods=['POST'])
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
