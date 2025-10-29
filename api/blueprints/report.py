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
    current_request_time = datetime.now()

    if 'files' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('files')
    record_id = request.form.get('record_id', type=int)
    exercise_id = request.form.get('exercise_id', type=int)

    # 删除了 is_waiting_for_feedback 和 P 的所有逻辑
    
    if not record_id or not exercise_id:
        return jsonify({'error': 'Fields "record_id" and "exercise_id" are required'}), 400

    record_detail = RecoveryRecordDetail.query.filter_by(record_id=record_id, exercise_id=exercise_id).first()
    
    if not record_detail:
        record_detail = RecoveryRecordDetail(record_id=record_id, exercise_id=exercise_id)
        db.session.add(record_detail)
        record = RecoveryRecord.query.get(record_id)
        if not record: return jsonify({'error': f'Record with id {record_id} not found.'}), 404
        
        record_detail.last_feedback_timestamp = current_request_time
        
        try:
            db.session.flush()
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Failed to create new RecordDetail: {e}")
            return jsonify({'error': 'Failed to initialize record detail'}), 500

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
            current_app.logger.error(f"Error predicting frames for {filename}: {e}", exc_info=True)
            frame_results = [0] * 6 

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

    response = {
        'message': f'Successfully uploaded {len(saved_files_info)} sprite sheets for exercise {exercise_id}.',
        'files_saved': saved_files_info,
        'exercise_id': exercise_id
    }
        
    return jsonify(response), 201


@report_bp.route('/reports/feedback', methods=['GET'])
def get_feedback():
    current_request_time = datetime.now()
    record_id = request.args.get('record_id', type=int)
    exercise_id = request.args.get('exercise_id', type=int)

    if not record_id or not exercise_id:
        return jsonify({'error': 'Query parameters "record_id" and "exercise_id" are required'}), 400

    exercise = Exercise.query.get(exercise_id)
    if not exercise or not exercise.duration_seconds:
        return jsonify({'error': 'Exercise not found or duration not set'}), 404
    T = exercise.duration_seconds
    P = (T + 6) / 4
    if P < 8: P *= 2
    elif P > 20: P /= 2
    
    record_detail = RecoveryRecordDetail.query.filter_by(record_id=record_id, exercise_id=exercise_id).first()
    
    if not record_detail:
        current_app.logger.warning(f"get_feedback called for non-existent RecordDetail (rec:{record_id}, ex:{exercise_id}). Creating.")
        record_detail = RecoveryRecordDetail(record_id=record_id, exercise_id=exercise_id)
        db.session.add(record_detail)
        record = RecoveryRecord.query.get(record_id)
        if not record: return jsonify({'error': f'Record with id {record_id} not found.'}), 404
        record_detail.last_feedback_timestamp = current_request_time
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': 'Failed to initialize record detail'}), 500
            
    record = record_detail.recovery_record
    elapsed_duration_for_response = (current_request_time - record.record_date).total_seconds()

    feedback_response = {}
    should_generate_feedback = False
    start_window = None
    end_window = current_request_time

    try:
        record_detail_locked = RecoveryRecordDetail.query.with_for_update().filter_by(record_id=record_id, exercise_id=exercise_id).first()
        
        if not record_detail_locked:
            raise Exception("RecordDetail mysteriously disappeared")

        last_feedback_time = record_detail_locked.last_feedback_timestamp
        
        if not last_feedback_time:
            last_feedback_time = record_detail_locked.recovery_record.record_date

        if current_request_time >= last_feedback_time + timedelta(seconds=P):
            should_generate_feedback = True
            start_window = last_feedback_time 
            
            record_detail_locked.last_feedback_timestamp = current_request_time
            
            db.session.commit()
        else:
            db.session.rollback()
            should_generate_feedback = False

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Database lock/check failed during feedback decision: {e}")
        should_generate_feedback = False
        return jsonify({'feedback': None, 'exercise_id': exercise_id, 'message': 'Lock contention or DB error'}), 503
    
    if should_generate_feedback:
        try:
            feedback_text = current_app.report_service.generate_slice_video_feedback(
                record_id, exercise_id, start_window, end_window
            )
            
            feedback_response = {'feedback': feedback_text, 'timestamp': elapsed_duration_for_response}

            record_detail_to_update = RecoveryRecordDetail.query.get(record_detail_locked.record_detail_id)
            existing_evaluations = record_detail_to_update.slice_evaluations or []
            existing_evaluations.append(feedback_response)
            record_detail_to_update.slice_evaluations = existing_evaluations
            
            db.session.commit()
            
            return jsonify({
                'feedback': feedback_text,
                'exercise_id': exercise_id,
                'timestamp': elapsed_duration_for_response
            }), 200
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Feedback generation or saving failed: {e}")
            return jsonify({'feedback': None, 'exercise_id': exercise_id, 'message': f'Feedback generation failed: {e}'}), 500
    
    return jsonify({'feedback': None, 'exercise_id': exercise_id}), 200

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