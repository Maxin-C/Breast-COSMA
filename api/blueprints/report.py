import os
import json
import time
from types import SimpleNamespace
import numpy as np
import cv2
from datetime import datetime, timedelta
from dotenv import load_dotenv
from PIL import Image

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from api.extensions import db
from utils.database.models import VideoSliceImage, RecoveryRecord, RecoveryRecordDetail, Exercise

load_dotenv()

report_bp = Blueprint('report', __name__)

UPLOAD_FOLDER = os.getenv("SLICE_SAVE_PATH", "uploads/slices")
VIDEO_FOLDER = os.getenv("VIDEO_SAVE_PATH", "uploads/videos")

@report_bp.route('/detect_upper_body', methods=['POST'])
def detect_upper_body():
    # ... (this function remains the same)
    detector = current_app.upper_body_detector
    if detector is None:
        return jsonify({"error": "Upper body detector is not available."}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image."}), 400

        current_detection = detector.detect(img)
        
        return jsonify({"is_upper_body_in_frame": current_detection}), 200

    except Exception as e:
        current_app.logger.error(f"Error during detection: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@report_bp.route('/reports/upload_sprites', methods=['POST'])
def upload_sprite_sheet():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('files')
    record_id = request.form.get('record_id', type=int)
    exercise_id = request.form.get('exercise_id', type=int)
    elapsed_time = request.form.get('elapsed_time', type=float)
    is_waiting_for_feedback = request.form.get('is_waiting_for_feedback', 'false').lower() == 'true'

    if not record_id or not exercise_id or elapsed_time is None:
        return jsonify({'error': 'Fields "record_id", "exercise_id", and "elapsed_time" are required'}), 400

    exercise = Exercise.query.get(exercise_id)
    if not exercise or not exercise.duration_seconds:
        return jsonify({'error': 'Exercise not found or duration not set'}), 404
    T = exercise.duration_seconds

    P = (T - 4) / 4
    if P < 8: P *= 2
    elif P > 20: P /= 2
    
    record_detail = RecoveryRecordDetail.query.filter_by(record_id=record_id, exercise_id=exercise_id).first()
    if not record_detail:
        record_detail = RecoveryRecordDetail(record_id=record_id, exercise_id=exercise_id)
        db.session.add(record_detail)
        record = RecoveryRecord.query.get(record_id)
        if not record: return jsonify({'error': f'Record with id {record_id} not found.'}), 404
        record_detail.last_feedback_timestamp = record.record_date
        db.session.flush()

    record = record_detail.recovery_record
    last_feedback_time = record_detail.last_feedback_timestamp or record.record_date
    current_time = record.record_date + timedelta(seconds=elapsed_time)

    should_feedback = False
    if not is_waiting_for_feedback and current_time >= last_feedback_time + timedelta(seconds=P):
        should_feedback = True

    upload_path = UPLOAD_FOLDER
    os.makedirs(upload_path, exist_ok=True)
    saved_files_info = []
    slice_order_start = db.session.query(db.func.max(VideoSliceImage.slice_order)).filter_by(record_id=record_id, exercise_id=exercise_id).scalar() or 0

    for i, file in enumerate(files, 1):
        filename = secure_filename(f"rec{record_id}_ex{exercise_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{i}.jpg")
        absolute_filepath = os.path.join(upload_path, filename)
        file.save(absolute_filepath)
        
        db_relative_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            pil_image = Image.open(absolute_filepath)
            frame_results = current_app.action_classifier_service.predict_frames_in_sprite(
                pil_image,
                int(exercise_id)
            )
        except Exception as e:
            # If model prediction fails, log it but don't stop the upload process
            current_app.logger.error(f"Error predicting frames for {filename}: {e}", exc_info=True)
            frame_results = [0] * 6 # Assume not part of action if prediction fails

        slice_image = VideoSliceImage(
            record_id=record_id,
            exercise_id=int(exercise_id),
            slice_order=slice_order_start + i,
            image_path=db_relative_path,
            timestamp=datetime.now(),
            is_part_of_action=(all(x == 1 for x in frame_results))
        )
        db.session.add(slice_image)
        saved_files_info.append(db_relative_path)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Database commit failed during sprite saving: {e}")
        return jsonify({'error': f'Failed to save image records to database: {str(e)}'}), 500

    feedback_response = {}
    if should_feedback:
        start_window = last_feedback_time
        end_window = current_time
        
        feedback_text = current_app.report_service.generate_slice_video_feedback(
            record_id, exercise_id, start_window, end_window
        )
        feedback_response = {'feedback': feedback_text, 'timestamp': elapsed_time}

        record_detail.last_feedback_timestamp = current_time
        
        existing_evaluations = record_detail.slice_evaluations or []
        existing_evaluations.append(feedback_response)
        record_detail.slice_evaluations = existing_evaluations
        
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Database commit failed during feedback saving: {e}")
            feedback_response = {'feedback': '保存反馈时出错。', 'timestamp': elapsed_time}
    
    response = {
        'message': f'Successfully uploaded {len(saved_files_info)} sprite sheets for exercise {exercise_id}.',
        'files_saved': saved_files_info
    }
    if feedback_response:
        response.update(feedback_response)
        
    return jsonify(response), 201

# ... (rest of the file remains the same)
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
        
@report_bp.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)