from flask import Flask, request, jsonify
from config import Config
from utils.database.models import db, User, RecoveryPlan, Exercise, UserRecoveryPlan, CalendarSchedule, RecoveryRecord, RecoveryRecordDetail, MessageChat, VideoSliceImage, Form, QoL
from utils.database.forms import (
    UserForm, RecoveryPlanForm, ExerciseForm, UserRecoveryPlanForm,
    CalendarScheduleForm, RecoveryRecordForm, RecoveryRecordDetailForm,
    MessageChatForm, VideoSliceImageForm, FormForm, QoLForm
)
from utils.database import database as db_operations
from utils.detect_upper_body.main import UpperBodyDetector

from flask_compress import Compress
from flask_wtf.csrf import CSRFProtect
import secrets
import os
from types import SimpleNamespace
import json
import numpy as np
import cv2

app = Flask(__name__)
Compress(app)
app.config.from_object(Config)
# app.config['SECRET_KEY'] = secrets.token_hex(16)
# app.config['SECRET_KEY'] = "test"
# csrf = CSRFProtect(app)
app.config['WTF_CSRF_ENABLED'] = False
db.init_app(app)

#--------------------------------------------------
# upper body detector
mmpose_config_path="utils/pose_estimation/mmpose_config.json"
mmpose_config = SimpleNamespace(**json.load(open(mmpose_config_path,'r')))
detector_device = "cuda:0"
try:
    detector = UpperBodyDetector(mmpose_config.pose2d_config, mmpose_config.pose2d_checkpoint, confidence_threshold=0.5, device=detector_device)
except Exception as e:
    print(f"Flask 应用启动失败：无法初始化 UpperBodyDetector。错误：{e}")
    exit(1)

@app.route('/detect_upper_body', methods=['POST'])
def detect_upper_body():
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

            response_data = {"is_upper_body_in_frame": detector.detect(img)}
            return jsonify(response_data), 200

        except Exception as e:
            print(f"Error during detection: {e}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# --------------------------------------------------
# video
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

@app.route('/static/<path:filename>')
def serve_static(filename):
    # send_from_directory 会自动处理文件类型和 Content-Disposition
    return send_from_directory(STATIC_FOLDER, filename)

# --------------------------------------------------
# dataset

# Create database tables if they don't exist (run once)
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    """Home page."""
    return jsonify({"message": "Welcome to the Breast Cosma Database API! Use specific endpoints for CRUD operations."})

# --- CRUD Operations for Users ---
@app.route('/users', methods=['GET'])
def list_users():
    """List all users."""
    users = db_operations.get_all_records(User)
    return jsonify([user.to_dict() for user in users])

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get a single user by ID."""
    user = db_operations.get_record_by_id(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404
    return jsonify(user.to_dict())

# 新增：根据字段查询用户
@app.route('/users/search', methods=['GET'])
def search_users():
    """
    Search users by a specified field and value.
    Example: /users/search?field=name&value=John Doe
    """
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    # 简单处理类型转换，例如尝试将数字字符串转换为整数
    # 更严谨的做法是根据模型字段类型进行转换
    try:
        # 尝试转换为整数，如果字段名是ID类
        if field_name.endswith('_id') or field_name == 'srrsh_id':
            field_value = int(field_value)
    except ValueError:
        # 如果转换失败，保留为字符串，让数据库处理
        pass

    users = db_operations.get_records_by_field(User, field_name, field_value)
    if users:
        return jsonify([user.to_dict() for user in users])
    else:
        return jsonify({"message": f"No users found with {field_name} = {field_value}"}), 404

@app.route('/users', methods=['POST'])
def add_user():
    """Add a new user."""
    # Ensure the request is JSON format
    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form = UserForm(data=request.json) # Directly load data from request.json

    if form.validate(): # For JSON API, directly call form.validate()
        user_data = {field.name: field.data for field in form if field.name != 'csrf_token'}

        # If user_id is auto-incrementing, remove it from user_data
        # You might want to uncomment this if your user_id is truly auto-incrementing
        user_data.pop('user_id', None)

        new_user = db_operations.add_record(User, user_data)
        if new_user:
            return jsonify({"message": "User added successfully!", "user": new_user.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding user."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/users/<int:user_id>', methods=['PUT'])
def edit_user(user_id):
    """Edit an existing user."""
    user = db_operations.get_record_by_id(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404

    # Ensure the request is JSON format
    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form = UserForm(data=request.json, obj=user) # Load data from JSON, populate existing object

    if form.validate(): # For JSON API, directly call form.validate()
        user_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        # Remove user_id from update data if it's an auto-incrementing primary key
        user_data.pop('user_id', None) # Primary keys are usually not updated

        updated_user = db_operations.update_record(user, user_data)
        if updated_user:
            return jsonify({"message": "User updated successfully!", "user": updated_user.to_dict()})
        else:
            return jsonify({"message": "Error updating user."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user."""
    user = db_operations.get_record_by_id(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404

    if db_operations.delete_record(user):
        return jsonify({"message": "User deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting user."}), 500

# --- CRUD Operations for Recovery Plans ---
@app.route('/recovery_plans', methods=['GET'])
def list_recovery_plans():
    """List all recovery plans."""
    plans = db_operations.get_all_records(RecoveryPlan)
    return jsonify([plan.to_dict() for plan in plans])

@app.route('/recovery_plans/<int:plan_id>', methods=['GET'])
def get_recovery_plan(plan_id):
    """Get a single recovery plan by ID."""
    plan = db_operations.get_record_by_id(RecoveryPlan, plan_id)
    if not plan:
        return jsonify({"message": "Recovery plan not found."}), 404
    return jsonify(plan.to_dict())

# 新增：根据字段查询恢复计划
@app.route('/recovery_plans/search', methods=['GET'])
def search_recovery_plans():
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id'):
            field_value = int(field_value)
        elif field_name.endswith('_date'): # 处理日期字段
            field_value = datetime.strptime(field_value, '%Y-%m-%d').date()
    except ValueError:
        pass # If conversion fails, keep as string

    plans = db_operations.get_records_by_field(RecoveryPlan, field_name, field_value)
    if plans:
        return jsonify([plan.to_dict() for plan in plans])
    else:
        return jsonify({"message": f"No recovery plans found with {field_name} = {field_value}"}), 404

@app.route('/recovery_plans', methods=['POST'])
def add_recovery_plan():
    """Add a new recovery plan."""
    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form = RecoveryPlanForm(data=request.json)

    if form.validate():
        plan_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        plan_data.pop('plan_id', None) # Assume plan_id is auto-incrementing

        new_plan = db_operations.add_record(RecoveryPlan, plan_data)
        if new_plan:
            return jsonify({"message": "Recovery plan added successfully!", "plan": new_plan.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding recovery plan."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/recovery_plans/<int:plan_id>', methods=['PUT'])
def edit_recovery_plan(plan_id):
    """Edit an existing recovery plan."""
    plan = db_operations.get_record_by_id(RecoveryPlan, plan_id)
    if not plan:
        return jsonify({"message": "Recovery plan not found."}), 404

    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form = RecoveryPlanForm(data=request.json, obj=plan)

    if form.validate():
        plan_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        plan_data.pop('plan_id', None)

        updated_plan = db_operations.update_record(plan, plan_data)
        if updated_plan:
            return jsonify({"message": "Recovery plan updated successfully!", "plan": updated_plan.to_dict()})
        else:
            return jsonify({"message": "Error updating recovery plan."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/recovery_plans/<int:plan_id>', methods=['DELETE'])
def delete_recovery_plan(plan_id):
    """Delete a recovery plan."""
    plan = db_operations.get_record_by_id(RecoveryPlan, plan_id)
    if not plan:
        return jsonify({"message": "Recovery plan not found."}), 404

    if db_operations.delete_record(plan):
        return jsonify({"message": "Recovery plan deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting recovery plan."}), 500

# --- CRUD Operations for Exercises ---
@app.route('/exercises', methods=['GET'])
def list_exercises():
    exercises = db_operations.get_all_records(Exercise)
    return jsonify([exercise.to_dict() for exercise in exercises])

@app.route('/exercises/<int:exercise_id>', methods=['GET'])
def get_exercise(exercise_id):
    exercise = db_operations.get_record_by_id(Exercise, exercise_id)
    if not exercise:
        return jsonify({"message": "Exercise not found."}), 404
    return jsonify(exercise.to_dict())

# 新增：根据字段查询练习
@app.route('/exercises/search', methods=['GET'])
def search_exercises():
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id') or field_name.startswith('duration') or field_name == 'repetitions':
            field_value = int(field_value)
    except ValueError:
        pass

    exercises = db_operations.get_records_by_field(Exercise, field_name, field_value)
    if exercises:
        return jsonify([exercise.to_dict() for exercise in exercises])
    else:
        return jsonify({"message": f"No exercises found with {field_name} = {field_value}"}), 404

@app.route('/exercises', methods=['POST'])
def add_exercise():
    form = ExerciseForm(request.form)
    if not form.validate_on_submit() and request.json:
        form = ExerciseForm(data=request.json)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        exercise_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        exercise_data.pop('exercise_id', None) # Assume exercise_id is auto-incrementing

        new_exercise = db_operations.add_record(Exercise, exercise_data)
        if new_exercise:
            return jsonify({"message": "Exercise added successfully!", "exercise": new_exercise.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding exercise."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/exercises/<int:exercise_id>', methods=['PUT'])
def edit_exercise(exercise_id):
    exercise = db_operations.get_record_by_id(Exercise, exercise_id)
    if not exercise:
        return jsonify({"message": "Exercise not found."}), 404

    form = ExerciseForm(request.form, obj=exercise)
    if not form.validate_on_submit() and request.json:
        form = ExerciseForm(data=request.json, obj=exercise)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        exercise_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        exercise_data.pop('exercise_id', None)

        updated_exercise = db_operations.update_record(exercise, exercise_data)
        if updated_exercise:
            return jsonify({"message": "Exercise updated successfully!", "exercise": updated_exercise.to_dict()})
        else:
            return jsonify({"message": "Error updating exercise."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/exercises/<int:exercise_id>', methods=['DELETE'])
def delete_exercise(exercise_id):
    exercise = db_operations.get_record_by_id(Exercise, exercise_id)
    if not exercise:
        return jsonify({"message": "Exercise not found."}), 404

    if db_operations.delete_record(exercise):
        return jsonify({"message": "Exercise deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting exercise."}), 500

# --- CRUD Operations for User Recovery Plans ---
@app.route('/user_recovery_plans', methods=['GET'])
def list_user_recovery_plans():
    user_recovery_plans = db_operations.get_all_records(UserRecoveryPlan)
    return jsonify([plan.to_dict() for plan in user_recovery_plans])

@app.route('/user_recovery_plans/<int:user_plan_id>', methods=['GET'])
def get_user_recovery_plan(user_plan_id):
    plan = db_operations.get_record_by_id(UserRecoveryPlan, user_plan_id)
    if not plan:
        return jsonify({"message": "User Recovery Plan not found."}), 404
    return jsonify(plan.to_dict())

# 新增：根据字段查询用户恢复计划
@app.route('/user_recovery_plans/search', methods=['GET'])
def search_user_recovery_plans():
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id'):
            field_value = int(field_value)
        elif field_name.endswith('_date'):
            field_value = datetime.strptime(field_value, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass

    plans = db_operations.get_records_by_field(UserRecoveryPlan, field_name, field_value)
    if plans:
        return jsonify([plan.to_dict() for plan in plans])
    else:
        return jsonify({"message": f"No user recovery plans found with {field_name} = {field_value}"}), 404

@app.route('/user_recovery_plans', methods=['POST'])
def add_user_recovery_plan():
    form = UserRecoveryPlanForm(request.form)
    if not form.validate_on_submit() and request.json:
        form = UserRecoveryPlanForm(data=request.json)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('user_plan_id', None)

        new_record = db_operations.add_record(UserRecoveryPlan, data)
        if new_record:
            return jsonify({"message": "User Recovery Plan added successfully!", "user_plan": new_record.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding User Recovery Plan."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/user_recovery_plans/<int:user_plan_id>', methods=['PUT'])
def edit_user_recovery_plan(user_plan_id):
    record = db_operations.get_record_by_id(UserRecoveryPlan, user_plan_id)
    if not record:
        return jsonify({"message": "User Recovery Plan not found."}), 404

    form = UserRecoveryPlanForm(request.form, obj=record)
    if not form.validate_on_submit() and request.json:
        form = UserRecoveryPlanForm(data=request.json, obj=record)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('user_plan_id', None)

        updated_record = db_operations.update_record(record, data)
        if updated_record:
            return jsonify({"message": "User Recovery Plan updated successfully!", "user_plan": updated_record.to_dict()})
        else:
            return jsonify({"message": "Error updating User Recovery Plan."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/user_recovery_plans/<int:user_plan_id>', methods=['DELETE'])
def delete_user_recovery_plan(user_plan_id):
    record = db_operations.get_record_by_id(UserRecoveryPlan, user_plan_id)
    if not record:
        return jsonify({"message": "User Recovery Plan not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "User Recovery Plan deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting User Recovery Plan."}), 500

# --- CRUD Operations for Calendar Schedules ---
@app.route('/calendar_schedules', methods=['GET'])
def list_calendar_schedules():
    schedules = db_operations.get_all_records(CalendarSchedule)
    return jsonify([schedule.to_dict() for schedule in schedules])

@app.route('/calendar_schedules/<int:schedule_id>', methods=['GET'])
def get_calendar_schedule(schedule_id):
    schedule = db_operations.get_record_by_id(CalendarSchedule, schedule_id)
    if not schedule:
        return jsonify({"message": "Calendar Schedule not found."}), 404
    return jsonify(schedule.to_dict())

# 新增：根据字段查询日历日程
@app.route('/calendar_schedules/search', methods=['GET'])
def search_calendar_schedules():
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id'):
            field_value = int(field_value)
        elif field_name.endswith('_date'):
            field_value = datetime.strptime(field_value, '%Y-%m-%d').date()
        elif field_name.endswith('_time'):
            field_value = datetime.strptime(field_value, '%H:%M:%S').time()
        elif field_name == 'is_completed':
            field_value = field_value.lower() == 'true'
    except ValueError:
        pass

    schedules = db_operations.get_records_by_field(CalendarSchedule, field_name, field_value)
    if schedules:
        return jsonify([schedule.to_dict() for schedule in schedules])
    else:
        return jsonify({"message": f"No calendar schedules found with {field_name} = {field_value}"}), 404


@app.route('/calendar_schedules', methods=['POST'])
def add_calendar_schedule():
    form = CalendarScheduleForm(request.form)
    if not form.validate_on_submit() and request.json:
        form = CalendarScheduleForm(data=request.json)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('schedule_id', None)

        new_record = db_operations.add_record(CalendarSchedule, data)
        if new_record:
            return jsonify({"message": "Calendar Schedule added successfully!", "schedule": new_record.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding Calendar Schedule."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/calendar_schedules/<int:schedule_id>', methods=['PUT'])
def edit_calendar_schedule(schedule_id):
    record = db_operations.get_record_by_id(CalendarSchedule, schedule_id)
    if not record:
        return jsonify({"message": "Calendar Schedule not found."}), 404

    form = CalendarScheduleForm(request.form, obj=record)
    if not form.validate_on_submit() and request.json:
        form = CalendarScheduleForm(data=request.json, obj=record)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('schedule_id', None)

        updated_record = db_operations.update_record(record, data)
        if updated_record:
            return jsonify({"message": "Calendar Schedule updated successfully!", "schedule": updated_record.to_dict()})
        else:
            return jsonify({"message": "Error updating Calendar Schedule."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/calendar_schedules/<int:schedule_id>', methods=['DELETE'])
def delete_calendar_schedule(schedule_id):
    record = db_operations.get_record_by_id(CalendarSchedule, schedule_id)
    if not record:
        return jsonify({"message": "Calendar Schedule not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Calendar Schedule deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Calendar Schedule."}), 500

# --- CRUD Operations for Recovery Records ---
@app.route('/recovery_records', methods=['GET'])
def list_recovery_records():
    records = db_operations.get_all_records(RecoveryRecord)
    return jsonify([record.to_dict() for record in records])

@app.route('/recovery_records/<int:record_id>', methods=['GET'])
def get_recovery_record(record_id):
    record = db_operations.get_record_by_id(RecoveryRecord, record_id)
    if not record:
        return jsonify({"message": "Recovery Record not found."}), 404
    return jsonify(record.to_dict())

# 新增：根据字段查询恢复记录
@app.route('/recovery_records/search', methods=['GET'])
def search_recovery_records():
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id'):
            field_value = int(field_value)
        elif field_name.endswith('_date'):
            field_value = datetime.strptime(field_value, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass

    records = db_operations.get_records_by_field(RecoveryRecord, field_name, field_value)
    if records:
        return jsonify([record.to_dict() for record in records])
    else:
        return jsonify({"message": f"No recovery records found with {field_name} = {field_value}"}), 404


@app.route('/recovery_records', methods=['POST'])
def add_recovery_record():
    form = RecoveryRecordForm(request.form)
    if not form.validate_on_submit() and request.json:
        form = RecoveryRecordForm(data=request.json)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('record_id', None)

        new_record = db_operations.add_record(RecoveryRecord, data)
        if new_record:
            return jsonify({"message": "Recovery Record added successfully!", "record": new_record.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding Recovery Record."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/recovery_records/<int:record_id>', methods=['PUT'])
def edit_recovery_record(record_id):
    record = db_operations.get_record_by_id(RecoveryRecord, record_id)
    if not record:
        return jsonify({"message": "Recovery Record not found."}), 404

    form = RecoveryRecordForm(request.form, obj=record)
    if not form.validate_on_submit() and request.json:
        form = RecoveryRecordForm(data=request.json, obj=record)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('record_id', None)

        updated_record = db_operations.update_record(record, data)
        if updated_record:
            return jsonify({"message": "Recovery Record updated successfully!", "record": updated_record.to_dict()})
        else:
            return jsonify({"message": "Error updating Recovery Record."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/recovery_records/<int:record_id>', methods=['DELETE'])
def delete_recovery_record(record_id):
    record = db_operations.get_record_by_id(RecoveryRecord, record_id)
    if not record:
        return jsonify({"message": "Recovery Record not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Recovery Record deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Recovery Record."}), 500

# --- CRUD Operations for Recovery Record Details ---
@app.route('/recovery_record_details', methods=['GET'])
def list_recovery_record_details():
    details = db_operations.get_all_records(RecoveryRecordDetail)
    return jsonify([detail.to_dict() for detail in details])

@app.route('/recovery_record_details/<int:record_detail_id>', methods=['GET'])
def get_recovery_record_detail(record_detail_id):
    detail = db_operations.get_record_by_id(RecoveryRecordDetail, record_detail_id)
    if not detail:
        return jsonify({"message": "Recovery Record Detail not found."}), 404
    return jsonify(detail.to_dict())

# 新增：根据字段查询恢复记录详情
@app.route('/recovery_record_details/search', methods=['GET'])
def search_recovery_record_details():
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id') or field_name.startswith('actual_'):
            field_value = int(field_value)
        elif field_name == 'completion_timestamp':
            field_value = datetime.strptime(field_value, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass

    details = db_operations.get_records_by_field(RecoveryRecordDetail, field_name, field_value)
    if details:
        return jsonify([detail.to_dict() for detail in details])
    else:
        return jsonify({"message": f"No recovery record details found with {field_name} = {field_value}"}), 404


@app.route('/recovery_record_details', methods=['POST'])
def add_recovery_record_detail():
    form = RecoveryRecordDetailForm(request.form)
    if not form.validate_on_submit() and request.json:
        form = RecoveryRecordDetailForm(data=request.json)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        # record_detail_id is autoincrement, so it's Optional for add. No need to pop if not present
        # data.pop('record_detail_id', None) # If you explicitly send it and want it ignored

        new_record = db_operations.add_record(RecoveryRecordDetail, data)
        if new_record:
            return jsonify({"message": "Recovery Record Detail added successfully!", "detail": new_record.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding Recovery Record Detail."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/recovery_record_details/<int:record_detail_id>', methods=['PUT'])
def edit_recovery_record_detail(record_detail_id):
    record = db_operations.get_record_by_id(RecoveryRecordDetail, record_detail_id)
    if not record:
        return jsonify({"message": "Recovery Record Detail not found."}), 404

    form = RecoveryRecordDetailForm(request.form, obj=record)
    if not form.validate_on_submit() and request.json:
        form = RecoveryRecordDetailForm(data=request.json, obj=record)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        # record_detail_id is autoincrement, usually not updated
        data.pop('record_detail_id', None)

        updated_record = db_operations.update_record(record, data)
        if updated_record:
            return jsonify({"message": "Recovery Record Detail updated successfully!", "detail": updated_record.to_dict()})
        else:
            return jsonify({"message": "Error updating Recovery Record Detail."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/recovery_record_details/<int:record_detail_id>', methods=['DELETE'])
def delete_recovery_record_detail(record_detail_id):
    record = db_operations.get_record_by_id(RecoveryRecordDetail, record_detail_id)
    if not record:
        return jsonify({"message": "Recovery Record Detail not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Recovery Record Detail deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Recovery Record Detail."}), 500

# --- CRUD Operations for Message Chats ---
@app.route('/messages_chat', methods=['GET'])
def list_messages_chat():
    messages = db_operations.get_all_records(MessageChat)
    return jsonify([message.to_dict() for message in messages])

@app.route('/messages_chat/<int:message_id>', methods=['GET'])
def get_message_chat(message_id):
    message = db_operations.get_record_by_id(MessageChat, message_id)
    if not message:
        return jsonify({"message": "Message Chat not found."}), 404
    return jsonify(message.to_dict())

# 新增：根据字段查询消息聊天
@app.route('/messages_chat/search', methods=['GET'])
def search_messages_chat():
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id'):
            field_value = int(field_value)
        elif field_name == 'timestamp':
            field_value = datetime.strptime(field_value, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass

    messages = db_operations.get_records_by_field(MessageChat, field_name, field_value)
    if messages:
        return jsonify([message.to_dict() for message in messages])
    else:
        return jsonify({"message": f"No message chats found with {field_name} = {field_value}"}), 404


@app.route('/messages_chat', methods=['POST'])
def add_message_chat():
    form = MessageChatForm(request.form)
    if not form.validate_on_submit() and request.json:
        form = MessageChatForm(data=request.json)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        # message_id is autoincrement, so it's Optional for add. No need to pop if not present
        # data.pop('message_id', None)

        new_record = db_operations.add_record(MessageChat, data)
        if new_record:
            return jsonify({"message": "Message Chat added successfully!", "chat": new_record.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding Message Chat."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/messages_chat/<int:message_id>', methods=['PUT'])
def edit_message_chat(message_id):
    record = db_operations.get_record_by_id(MessageChat, message_id)
    if not record:
        return jsonify({"message": "Message Chat not found."}), 404

    form = MessageChatForm(request.form, obj=record)
    if not form.validate_on_submit() and request.json:
        form = MessageChatForm(data=request.json, obj=record)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('message_id', None)

        updated_record = db_operations.update_record(record, data)
        if updated_record:
            return jsonify({"message": "Message Chat updated successfully!", "chat": updated_record.to_dict()})
        else:
            return jsonify({"message": "Error updating Message Chat."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/messages_chat/<int:message_id>', methods=['DELETE'])
def delete_message_chat(message_id):
    record = db_operations.get_record_by_id(MessageChat, message_id)
    if not record:
        return jsonify({"message": "Message Chat not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Message Chat deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Message Chat."}), 500

# --- CRUD Operations for Video Slice Images ---
@app.route('/video_slice_images', methods=['GET'])
def list_video_slice_images():
    images = db_operations.get_all_records(VideoSliceImage)
    return jsonify([image.to_dict() for image in images])

@app.route('/video_slice_images/<int:image_id>', methods=['GET'])
def get_video_slice_image(image_id):
    image = db_operations.get_record_by_id(VideoSliceImage, image_id)
    if not image:
        return jsonify({"message": "Video Slice Image not found."}), 404
    return jsonify(image.to_dict())

# 新增：根据字段查询视频切片图像
@app.route('/video_slice_images/search', methods=['GET'])
def search_video_slice_images():
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id') or field_name == 'slice_order':
            field_value = int(field_value)
        elif field_name == 'timestamp':
            field_value = datetime.strptime(field_value, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass

    images = db_operations.get_records_by_field(VideoSliceImage, field_name, field_value)
    if images:
        return jsonify([image.to_dict() for image in images])
    else:
        return jsonify({"message": f"No video slice images found with {field_name} = {field_value}"}), 404

@app.route('/video_slice_images', methods=['POST'])
def add_video_slice_image():
    form = VideoSliceImageForm(request.form)
    if not form.validate_on_submit() and request.json:
        form = VideoSliceImageForm(data=request.json)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('image_id', None)

        new_record = db_operations.add_record(VideoSliceImage, data)
        if new_record:
            return jsonify({"message": "Video Slice Image added successfully!", "image": new_record.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding Video Slice Image."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/video_slice_images/<int:image_id>', methods=['PUT'])
def edit_video_slice_image(image_id):
    record = db_operations.get_record_by_id(VideoSliceImage, image_id)
    if not record:
        return jsonify({"message": "Video Slice Image not found."}), 404

    form = VideoSliceImageForm(request.form, obj=record)
    if not form.validate_on_submit() and request.json:
        form = VideoSliceImageForm(data=request.json, obj=record)
        if not form.validate():
            return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    if form.validate_on_submit() or form.validate():
        data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        data.pop('image_id', None)

        updated_record = db_operations.update_record(record, data)
        if updated_record:
            return jsonify({"message": "Video Slice Image updated successfully!", "image": updated_record.to_dict()})
        else:
            return jsonify({"message": "Error updating Video Slice Image."}), 500
    return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/video_slice_images/<int:image_id>', methods=['DELETE'])
def delete_video_slice_image(image_id):
    record = db_operations.get_record_by_id(VideoSliceImage, image_id)
    if not record:
        return jsonify({"message": "Video Slice Image not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Video Slice Image deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Video Slice Image."}), 500

# --- CRUD Operations for Forms ---
@app.route('/forms', methods=['GET'])
def list_forms():
    """List all forms."""
    forms = db_operations.get_all_records(Form)
    return jsonify([form.to_dict() for form in forms])

@app.route('/forms/<int:form_id>', methods=['GET'])
def get_form(form_id):
    """Get a single form by ID."""
    form = db_operations.get_record_by_id(Form, form_id)
    if not form:
        return jsonify({"message": "Form not found."}), 404
    return jsonify(form.to_dict())

@app.route('/forms/search', methods=['GET'])
def search_forms():
    """
    Search forms by a specified field and value.
    Example: /forms/search?field=form_name&value=Daily Survey
    """
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id'):
            field_value = int(field_value)
    except ValueError:
        pass

    forms = db_operations.get_records_by_field(Form, field_name, field_value)
    if forms:
        return jsonify([form.to_dict() for form in forms])
    else:
        return jsonify({"message": f"No forms found with {field_name} = {field_value}"}), 404

@app.route('/forms', methods=['POST'])
def add_form():
    """Add a new form."""
    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form_data_from_request = request.json
    form = FormForm(data=form_data_from_request)

    if form.validate():
        new_form_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        new_form_data.pop('form_id', None) # Assume form_id is auto-incrementing

        new_form = db_operations.add_record(Form, new_form_data)
        if new_form:
            return jsonify({"message": "Form added successfully!", "form": new_form.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding form."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/forms/<int:form_id>', methods=['PUT'])
def edit_form(form_id):
    """Edit an existing form."""
    existing_form = db_operations.get_record_by_id(Form, form_id)
    if not existing_form:
        return jsonify({"message": "Form not found."}), 404

    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form_data_from_request = request.json
    form = FormForm(data=form_data_from_request, obj=existing_form)

    if form.validate():
        updated_form_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        updated_form_data.pop('form_id', None) # Primary keys are usually not updated

        updated_form = db_operations.update_record(existing_form, updated_form_data)
        if updated_form:
            return jsonify({"message": "Form updated successfully!", "form": updated_form.to_dict()})
        else:
            return jsonify({"message": "Error updating form."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/forms/<int:form_id>', methods=['DELETE'])
def delete_form(form_id):
    """Delete a form."""
    form = db_operations.get_record_by_id(Form, form_id)
    if not form:
        return jsonify({"message": "Form not found."}), 404

    if db_operations.delete_record(form):
        return jsonify({"message": "Form deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting form."}), 500

# --- CRUD Operations for QoL ---
@app.route('/qols', methods=['GET'])
def list_qols():
    """List all QoL records."""
    qols = db_operations.get_all_records(QoL)
    return jsonify([qol.to_dict() for qol in qols])

@app.route('/qols/<int:qol_id>', methods=['GET'])
def get_qol(qol_id):
    """Get a single QoL record by ID."""
    qol = db_operations.get_record_by_id(QoL, qol_id)
    if not qol:
        return jsonify({"message": "QoL record not found."}), 404
    return jsonify(qol.to_dict())

@app.route('/qols/search', methods=['GET'])
def search_qols():
    """
    Search QoL records by a specified field and value.
    Example: /qols/search?field=user_id&value=1
    """
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id') or field_name == 'score':
            field_value = int(field_value)
    except ValueError:
        pass

    qols = db_operations.get_records_by_field(QoL, field_name, field_value)
    if qols:
        return jsonify([qol.to_dict() for qol in qols])
    else:
        return jsonify({"message": f"No QoL records found with {field_name} = {field_value}"}), 404

@app.route('/qols', methods=['POST'])
def add_qol():
    """Add a new QoL record."""
    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form_data_from_request = request.json
    form = QoLForm(data=form_data_from_request)

    if form.validate():
        new_qol_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        new_qol_data.pop('qol_id', None) # Assume qol_id is auto-incrementing

        new_qol = db_operations.add_record(QoL, new_qol_data)
        if new_qol:
            return jsonify({"message": "QoL record added successfully!", "qol": new_qol.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding QoL record."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/qols/<int:qol_id>', methods=['PUT'])
def edit_qol(qol_id):
    """Edit an existing QoL record."""
    existing_qol = db_operations.get_record_by_id(QoL, qol_id)
    if not existing_qol:
        return jsonify({"message": "QoL record not found."}), 404

    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form_data_from_request = request.json
    form = QoLForm(data=form_data_from_request, obj=existing_qol)

    if form.validate():
        updated_qol_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        updated_qol_data.pop('qol_id', None) # Primary keys are usually not updated

        updated_qol = db_operations.update_record(existing_qol, updated_qol_data)
        if updated_qol:
            return jsonify({"message": "QoL record updated successfully!", "qol": updated_qol.to_dict()})
        else:
            return jsonify({"message": "Error updating QoL record."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@app.route('/qols/<int:qol_id>', methods=['DELETE'])
def delete_qol(qol_id):
    """Delete a QoL record."""
    qol = db_operations.get_record_by_id(QoL, qol_id)
    if not qol:
        return jsonify({"message": "QoL record not found."}), 404

    if db_operations.delete_record(qol):
        return jsonify({"message": "QoL record deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting QoL record."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)