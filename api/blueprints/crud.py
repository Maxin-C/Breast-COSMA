import os
import json
import numpy as np
import cv2
from flask import Blueprint, jsonify, request
from datetime import datetime, date, timedelta
from sqlalchemy import func, case, desc, and_
from dotenv import load_dotenv
import requests
load_dotenv()

from api.extensions import db
from utils.database import database as db_operations
from utils.database.models import (
    User, RecoveryPlan, Exercise, UserRecoveryPlan, CalendarSchedule, 
    RecoveryRecord, RecoveryRecordDetail, ChatHistory, VideoSliceImage, 
    QoL, Nurse, NurseEvaluation
)
from utils.database.forms import (
    UserForm, RecoveryPlanForm, ExerciseForm, UserRecoveryPlanForm,
    CalendarScheduleForm, RecoveryRecordForm, RecoveryRecordDetailForm,
    ChatHistoryForm, VideoSliceImageForm, QoLForm, NurseForm, NurseEvaluationForm
)
from utils.database import database as db_operations

crud_bp = Blueprint('crud', __name__)

def _calculate_plan_id(user):
    if not user.surgery_date:
        # 如果没有手术日期，默认分配第一个计划
        return 1

    # 计算术后天数，当天算第1天
    days_since_surgery = (date.today() - user.surgery_date.date()).days + 1
    
    if days_since_surgery <= 1:
        return 1
    elif days_since_surgery == 2:
        return 2
    elif days_since_surgery == 3:
        return 3
    elif 4 <= days_since_surgery <= 7:
        return 4
    else:  # 7天以后
        if user.extubation_status == '未拔管':
            return 5
        else:  # '已拔管'
            return 6

@crud_bp.route('/')
def index():
    return jsonify({"message": "Welcome to the Breast Cosma Database API! Use specific endpoints for CRUD operations."})

# --- CRUD Operations for Users ---
@crud_bp.route('/users', methods=['GET'])
def list_users():
    """List all users."""
    users = db_operations.get_all_records(User)
    return jsonify([user.to_dict() for user in users])

@crud_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get a single user by ID."""
    user = db_operations.get_record_by_id(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404
    return jsonify(user.to_dict())

@crud_bp.route('/users/search', methods=['GET'])
def search_users():
    """
    Search users by a specified field and value.
    Example: /users/search?field=name&value=John Doe
    """
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id') or field_name == 'srrsh_id':
            field_value = int(field_value)
    except ValueError:
        pass

    users = db_operations.get_records_by_field(User, field_name, field_value)
    if users:
        return jsonify([user.to_dict() for user in users])
    else:
        return jsonify({"message": f"No users found with {field_name} = {field_value}"}), 404

@crud_bp.route('/users/update_openid', methods=['POST'])
def update_user_openid():
    data = request.json
    user_id = data.get('user_id')
    code = data.get('code')

    if not user_id or not code:
        return jsonify({"error": "缺少 user_id 或 code"}), 400

    # 从应用配置中获取小程序的 appid 和 secret
    appid = os.getenv('WECHAT_APPID')
    secret = os.getenv('WECHAT_APPSECRET')

    if not appid or not secret:
        print("错误: WECHAT_APPID 或 WECHAT_APPSECRET 未在后端配置。")
        return jsonify({"error": "服务器配置错误"}), 500

    # 构造请求微信服务器的 URL
    url = f"https://api.weixin.qq.com/sns/jscode2session?appid={appid}&secret={secret}&js_code={code}&grant_type=authorization_code"

    try:
        # 发起请求
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败 (例如 4xx, 5xx), 则抛出异常
        wechat_data = response.json()

        openid = wechat_data.get('openid')
        if not openid:
            # 如果微信返回的数据中没有 openid，则返回错误
            print(f"从微信获取 openid 失败: {wechat_data}")
            return jsonify({"error": "无法从微信获取用户信息", "details": wechat_data}), 502 # 502 Bad Gateway

        # 从数据库中查找用户
        user = db_operations.get_record_by_id(User, user_id)
        if not user:
            return jsonify({"error": "指定的用户不存在"}), 404

        # 更新用户的 openid 并保存
        user.wechat_openid = openid
        db.session.commit()

        print(f"成功为用户 {user_id} 更新 OpenID。")
        return jsonify({"message": "用户信息同步成功"}), 200

    except requests.RequestException as e:
        print(f"请求微信服务器时发生网络错误: {e}")
        return jsonify({"error": "无法连接微信服务器"}), 503 # 503 Service Unavailable
    except Exception as e:
        db.session.rollback()
        print(f"更新 OpenID 时发生未知错误: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

@crud_bp.route('/users', methods=['POST'])
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

@crud_bp.route('/users/<int:user_id>', methods=['PUT'])
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

@crud_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user."""
    user = db_operations.get_record_by_id(User, user_id)
    if not user:
        return jsonify({"message": "User not found."}), 404

    if db_operations.delete_record(user):
        return jsonify({"message": "User deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting user."}), 500

@crud_bp.route('/users/register', methods=['POST'])
def register_user():
    """
    专门处理新用户注册的接口，包含护士编号验证和动态计划分配。
    """
    data = request.json
    name = data.get('name')
    surgery_date_str = data.get('surgery_date')
    nurse_id_suffix = data.get('nurse_id')

    if not all([name, surgery_date_str, nurse_id_suffix]):
        return jsonify({"error": "姓名、手术时间和护士编号均为必填项"}), 400

    # 步骤1: 验证护士编号
    nurse = db.session.query(Nurse).filter_by(phone_number_suffix=nurse_id_suffix).first()
    if not nurse:
        return jsonify({"error": "无效的护士编号，请核对后重试"}), 403

    # 步骤2: 检查用户是否已存在
    existing_user = db.session.query(User).filter_by(name=name).first()
    if existing_user:
        return jsonify({"error": f"姓名 '{name}' 已被注册，请直接登录"}), 409

    try:
        # 步骤3: 创建新用户
        new_user = User(
            name=name,
            surgery_date=datetime.strptime(surgery_date_str, '%Y-%m-%d'),
            extubation_status='未拔管',
            registration_date=datetime.utcnow()
        )
        db.session.add(new_user)
        db.session.flush()

        # 步骤4: 【核心修改】调用辅助函数计算正确的 plan_id
        new_plan_id = _calculate_plan_id(new_user)
        
        initial_plan = UserRecoveryPlan(
            user_id=new_user.user_id,
            plan_id=new_plan_id,
            status='active'
        )
        db.session.add(initial_plan)
        
        db.session.commit()

        return jsonify({
            "message": "用户注册成功！",
            "user": new_user.to_dict(),
            "user_plan": initial_plan.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        print(f"Error during user registration: {e}")
        return jsonify({"error": "服务器内部错误，注册失败"}), 500

@crud_bp.route('/users/login_logic', methods=['POST'])
def handle_login_logic():
    """
    处理登录后的业务逻辑：更新拔管状态和康复计划plan_id。
    """
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "缺少 user_id"}), 400

    user = db_operations.get_record_by_id(User, user_id)
    if not user:
        return jsonify({"error": "用户不存在"}), 404

    try:
        # 步骤1: 更新拔管状态
        new_extubation_status = data.get('extubation_status')
        if new_extubation_status and user.extubation_status != new_extubation_status:
            user.extubation_status = new_extubation_status
            print(f"User {user_id} extubation_status updated to {new_extubation_status}")

        # 步骤2: 【核心修改】调用辅助函数计算 plan_id
        new_plan_id = _calculate_plan_id(user)
        
        print(f"User {user_id}: New plan_id should be {new_plan_id}")

        # 步骤3: 更新或创建 UserRecoveryPlan
        user_plan = db.session.query(UserRecoveryPlan).filter_by(user_id=user_id).first()
        
        if user_plan:
            if user_plan.plan_id != new_plan_id:
                user_plan.plan_id = new_plan_id
                user_plan.assigned_date = datetime.utcnow()
                user_plan.status = 'active'
                print(f"User {user_id} recovery plan updated to plan_id {new_plan_id}")
        else:
            user_plan = UserRecoveryPlan(user_id=user_id, plan_id=new_plan_id, status='active')
            db.session.add(user_plan)
            print(f"New recovery plan created for user {user_id} with plan_id {new_plan_id}")

        db.session.commit()
        
        return jsonify({
            "message": "User status and plan updated successfully.",
            "user_plan_id": user_plan.user_plan_id,
            "plan_id": user_plan.plan_id
        }), 200

    except Exception as e:
        db.session.rollback()
        print(f"Error in handle_login_logic for user {user_id}: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

# --- CRUD Operations for Recovery Plans ---
@crud_bp.route('/recovery_plans', methods=['GET'])
def list_recovery_plans():
    """List all recovery plans."""
    plans = db_operations.get_all_records(RecoveryPlan)
    return jsonify([plan.to_dict() for plan in plans])

@crud_bp.route('/recovery_plans/<int:plan_id>', methods=['GET'])
def get_recovery_plan(plan_id):
    """Get a single recovery plan by ID."""
    plan = db_operations.get_record_by_id(RecoveryPlan, plan_id)
    if not plan:
        return jsonify({"message": "Recovery plan not found."}), 404
    return jsonify(plan.to_dict())

# 新增：根据字段查询恢复计划
@crud_bp.route('/recovery_plans/search', methods=['GET'])
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

@crud_bp.route('/recovery_plans', methods=['POST'])
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

@crud_bp.route('/recovery_plans/<int:plan_id>', methods=['PUT'])
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

@crud_bp.route('/recovery_plans/<int:plan_id>', methods=['DELETE'])
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
@crud_bp.route('/exercises', methods=['GET'])
def list_exercises():
    exercises = db_operations.get_all_records(Exercise)
    return jsonify([exercise.to_dict() for exercise in exercises])

@crud_bp.route('/exercises/<int:exercise_id>', methods=['GET'])
def get_exercise(exercise_id):
    exercise = db_operations.get_record_by_id(Exercise, exercise_id)
    if not exercise:
        return jsonify({"message": "Exercise not found."}), 404
    return jsonify(exercise.to_dict())

# 新增：根据字段查询练习
@crud_bp.route('/exercises/search', methods=['GET'])
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

@crud_bp.route('/exercises', methods=['POST'])
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

@crud_bp.route('/exercises/<int:exercise_id>', methods=['PUT'])
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

@crud_bp.route('/exercises/<int:exercise_id>', methods=['DELETE'])
def delete_exercise(exercise_id):
    exercise = db_operations.get_record_by_id(Exercise, exercise_id)
    if not exercise:
        return jsonify({"message": "Exercise not found."}), 404

    if db_operations.delete_record(exercise):
        return jsonify({"message": "Exercise deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting exercise."}), 500

# --- CRUD Operations for User Recovery Plans ---
@crud_bp.route('/user_recovery_plans', methods=['GET'])
def list_user_recovery_plans():
    user_recovery_plans = db_operations.get_all_records(UserRecoveryPlan)
    return jsonify([plan.to_dict() for plan in user_recovery_plans])

@crud_bp.route('/user_recovery_plans/<int:user_plan_id>', methods=['GET'])
def get_user_recovery_plan(user_plan_id):
    plan = db_operations.get_record_by_id(UserRecoveryPlan, user_plan_id)
    if not plan:
        return jsonify({"message": "User Recovery Plan not found."}), 404
    return jsonify(plan.to_dict())

# 新增：根据字段查询用户恢复计划
@crud_bp.route('/user_recovery_plans/search', methods=['GET'])
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

@crud_bp.route('/user_recovery_plans', methods=['POST'])
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

@crud_bp.route('/user_recovery_plans/<int:user_plan_id>', methods=['PUT'])
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

@crud_bp.route('/user_recovery_plans/<int:user_plan_id>', methods=['DELETE'])
def delete_user_recovery_plan(user_plan_id):
    record = db_operations.get_record_by_id(UserRecoveryPlan, user_plan_id)
    if not record:
        return jsonify({"message": "User Recovery Plan not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "User Recovery Plan deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting User Recovery Plan."}), 500

# --- CRUD Operations for Calendar Schedules ---
@crud_bp.route('/calendar_schedules', methods=['GET'])
def list_calendar_schedules():
    schedules = db_operations.get_all_records(CalendarSchedule)
    return jsonify([schedule.to_dict() for schedule in schedules])

@crud_bp.route('/calendar_schedules/<int:schedule_id>', methods=['GET'])
def get_calendar_schedule(schedule_id):
    schedule = db_operations.get_record_by_id(CalendarSchedule, schedule_id)
    if not schedule:
        return jsonify({"message": "Calendar Schedule not found."}), 404
    return jsonify(schedule.to_dict())

# 新增：根据字段查询日历日程
@crud_bp.route('/calendar_schedules/search', methods=['GET'])
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


@crud_bp.route('/calendar_schedules', methods=['POST'])
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

@crud_bp.route('/calendar_schedules/<int:schedule_id>', methods=['PUT'])
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

@crud_bp.route('/calendar_schedules/<int:schedule_id>', methods=['DELETE'])
def delete_calendar_schedule(schedule_id):
    record = db_operations.get_record_by_id(CalendarSchedule, schedule_id)
    if not record:
        return jsonify({"message": "Calendar Schedule not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Calendar Schedule deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Calendar Schedule."}), 500

# --- CRUD Operations for Recovery Records ---
@crud_bp.route('/recovery_records', methods=['GET'])
def list_recovery_records():
    records = db_operations.get_all_records(RecoveryRecord)
    return jsonify([record.to_dict() for record in records])

@crud_bp.route('/recovery_records/<int:record_id>', methods=['GET'])
def get_recovery_record(record_id):
    record = db_operations.get_record_by_id(RecoveryRecord, record_id)
    if not record:
        return jsonify({"message": "Recovery Record not found."}), 404
    return jsonify(record.to_dict())

# 新增：根据字段查询恢复记录
@crud_bp.route('/recovery_records/search', methods=['GET'])
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


@crud_bp.route('/recovery_records', methods=['POST'])
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

@crud_bp.route('/recovery_records/<int:record_id>', methods=['PUT'])
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

@crud_bp.route('/recovery_records/<int:record_id>', methods=['DELETE'])
def delete_recovery_record(record_id):
    record = db_operations.get_record_by_id(RecoveryRecord, record_id)
    if not record:
        return jsonify({"message": "Recovery Record not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Recovery Record deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Recovery Record."}), 500

# --- CRUD Operations for Recovery Record Details ---
@crud_bp.route('/recovery_record_details', methods=['GET'])
def list_recovery_record_details():
    details = db_operations.get_all_records(RecoveryRecordDetail)
    return jsonify([detail.to_dict() for detail in details])

@crud_bp.route('/recovery_record_details/<int:record_detail_id>', methods=['GET'])
def get_recovery_record_detail(record_detail_id):
    detail = db_operations.get_record_by_id(RecoveryRecordDetail, record_detail_id)
    if not detail:
        return jsonify({"message": "Recovery Record Detail not found."}), 404
    return jsonify(detail.to_dict())

# 新增：根据字段查询恢复记录详情
@crud_bp.route('/recovery_record_details/search', methods=['GET'])
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


@crud_bp.route('/recovery_record_details', methods=['POST'])
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

@crud_bp.route('/recovery_record_details/<int:record_detail_id>', methods=['PUT'])
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

@crud_bp.route('/recovery_record_details/<int:record_detail_id>', methods=['DELETE'])
def delete_recovery_record_detail(record_detail_id):
    record = db_operations.get_record_by_id(RecoveryRecordDetail, record_detail_id)
    if not record:
        return jsonify({"message": "Recovery Record Detail not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Recovery Record Detail deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Recovery Record Detail."}), 500

# --- CRUD Operations for Chat History ---

@crud_bp.route('/chat_history', methods=['GET'])
def list_chat_histories():
    """Lists all conversation histories."""
    histories = db_operations.get_all_records(ChatHistory)
    return jsonify([history.to_dict() for history in histories])

@crud_bp.route('/chat_history/<int:chat_id>', methods=['GET'])
def get_chat_history(chat_id):
    """Gets a specific conversation history by its primary key."""
    history = db_operations.get_record_by_id(ChatHistory, chat_id)
    if not history:
        return jsonify({"message": "Chat history not found."}), 404
    return jsonify(history.to_dict())

@crud_bp.route('/chat_history/search', methods=['GET'])
def search_chat_histories():
    """Searches for conversation histories by a specific field."""
    # This function remains largely the same, just uses the new model
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    # Basic type casting for search
    try:
        if field_name in ['user_id', 'message_id']:
            field_value = int(field_value)
    except ValueError:
        return jsonify({"message": f"Invalid value for field '{field_name}'."}), 400

    histories = db_operations.get_records_by_field(ChatHistory, field_name, field_value)
    if histories:
        return jsonify([history.to_dict() for history in histories])
    else:
        return jsonify({"message": f"No chat histories found with {field_name} = {field_value}"}), 404

@crud_bp.route('/chat_history', methods=['POST'])
def add_message_to_history():
    """
    Adds a new message. Creates a new history if conversation_id is new,
    or appends to an existing one.
    """
    form = ChatHistoryForm(data=request.json)
    if not form.validate():
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

    # Find if a conversation with this ID already exists
    existing_history = db_operations.get_records_by_field(
        ChatHistory, 'conversation_id', form.conversation_id.data
    )

    if existing_history:
        # --- Append to existing conversation ---
        record = existing_history[0]
        new_message = {'role': 'user', 'content': form.new_message_text.data}

        # SQLAlchemy's JSON type tracks changes, but it's safer to re-assign
        current_history = record.chat_history or []
        current_history.append(new_message)
        
        updated_data = {
            'chat_history': current_history,
            'is_follow_up': form.is_follow_up.data
        }

        updated_record = db_operations.update_record(record, updated_data)
        return jsonify({
            "message": "Message appended to existing chat history!",
            "chat": updated_record.to_dict()
        }), 200

    else:
        # --- Create new conversation history ---
        initial_history = [{'role': 'user', 'content': form.new_message_text.data}]
        
        data = {
            'conversation_id': form.conversation_id.data,
            'user_id': form.user_id.data,
            'is_follow_up': form.is_follow_up.data,
            'chat_history': initial_history
        }

        new_record = db_operations.add_record(ChatHistory, data)
        if new_record:
            return jsonify({
                "message": "New chat history created successfully!",
                "chat": new_record.to_dict()
            }), 201
        else:
            return jsonify({"message": "Error creating chat history."}), 500


@crud_bp.route('/chat_history/<int:chat_id>', methods=['PUT'])
def edit_chat_history_metadata(chat_id):
    """Updates metadata of a chat history (e.g., is_follow_up)."""
    record = db_operations.get_record_by_id(ChatHistory, chat_id)
    if not record:
        return jsonify({"message": "Chat history not found."}), 404

    # Assumes a simpler form for metadata updates, like ChatHistoryMetadataForm
    data = request.json
    if 'is_follow_up' not in data:
         return jsonify({"message": "Validation failed", "errors": "is_follow_up is required"}), 400

    update_data = {'is_follow_up': bool(data['is_follow_up'])}
    
    updated_record = db_operations.update_record(record, update_data)
    if updated_record:
        return jsonify({
            "message": "Chat history metadata updated successfully!",
            "chat": updated_record.to_dict()
        })
    else:
        return jsonify({"message": "Error updating chat history metadata."}), 500


@crud_bp.route('/chat_history/<int:chat_id>', methods=['DELETE'])
def delete_chat_history(chat_id):
    """Deletes an entire conversation history."""
    record = db_operations.get_record_by_id(ChatHistory, chat_id)
    if not record:
        return jsonify({"message": "Chat history not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Chat history deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting chat history."}), 500

@crud_bp.route('/chat_history/<string:conversation_id>/summarize', methods=['POST'])
def summarize_chat_history(conversation_id):
    """
    为指定的对话生成摘要并保存到数据库。
    """
    # 查找对话记录
    record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
    if not record:
        return jsonify({"message": "Chat history not found."}), 404

    try:
        consult_service = Consult()
        summary_text = consult_service.summarize_conversation(conversation_id)
        
        if summary_text:
            update_data = {'summary': summary_text}
            updated_record = db_operations.update_record(record, update_data)
            
            return jsonify({
                "message": "Summary generated and saved successfully!",
                "summary": updated_record.summary
            }), 200
        else:
            return jsonify({"message": "Failed to generate summary."}), 500

    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({"message": "An error occurred while generating the summary."}), 500

# --- CRUD Operations for Video Slice Images ---
@crud_bp.route('/video_slice_images', methods=['GET'])
def list_video_slice_images():
    images = db_operations.get_all_records(VideoSliceImage)
    return jsonify([image.to_dict() for image in images])

@crud_bp.route('/video_slice_images/<int:image_id>', methods=['GET'])
def get_video_slice_image(image_id):
    image = db_operations.get_record_by_id(VideoSliceImage, image_id)
    if not image:
        return jsonify({"message": "Video Slice Image not found."}), 404
    return jsonify(image.to_dict())

# 新增：根据字段查询视频切片图像
@crud_bp.route('/video_slice_images/search', methods=['GET'])
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

@crud_bp.route('/video_slice_images', methods=['POST'])
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

@crud_bp.route('/video_slice_images/<int:image_id>', methods=['PUT'])
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

@crud_bp.route('/video_slice_images/<int:image_id>', methods=['DELETE'])
def delete_video_slice_image(image_id):
    record = db_operations.get_record_by_id(VideoSliceImage, image_id)
    if not record:
        return jsonify({"message": "Video Slice Image not found."}), 404

    if db_operations.delete_record(record):
        return jsonify({"message": "Video Slice Image deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting Video Slice Image."}), 500

# --- CRUD Operations for QoL ---
@crud_bp.route('/qols', methods=['GET'])
def list_qols():
    """List all QoL records."""
    qols = db_operations.get_all_records(QoL)
    return jsonify([qol.to_dict() for qol in qols])

@crud_bp.route('/qols/<int:qol_id>', methods=['GET'])
def get_qol(qol_id):
    """Get a single QoL record by ID."""
    qol = db_operations.get_record_by_id(QoL, qol_id)
    if not qol:
        return jsonify({"message": "QoL record not found."}), 404
    return jsonify(qol.to_dict())

@crud_bp.route('/qols/search', methods=['GET'])
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
        if field_name.endswith('_id'):
            field_value = int(field_value)
    except ValueError:
        pass

    qols = db_operations.get_records_by_field(QoL, field_name, field_value)
    if qols:
        return jsonify([qol.to_dict() for qol in qols])
    else:
        return jsonify({"message": f"No QoL records found with {field_name} = {field_value}"}), 404

@crud_bp.route('/qols', methods=['POST'])
def add_qol():
    """Add a new QoL record."""
    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form_data_from_request = request.json
    
    # 处理result字段：如果是字符串，尝试解析为JSON
    if 'result' in form_data_from_request and isinstance(form_data_from_request['result'], str):
        try:
            form_data_from_request['result'] = json.loads(form_data_from_request['result'])
        except json.JSONDecodeError:
            return jsonify({"message": "Invalid JSON format in result field"}), 400
    
    form = QoLForm(data=form_data_from_request)

    if form.validate():
        new_qol_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        new_qol_data.pop('qol_id', None)  # Assume qol_id is auto-incrementing
        
        # 处理result字段
        if 'result' in new_qol_data and isinstance(new_qol_data['result'], str):
            try:
                new_qol_data['result'] = json.loads(new_qol_data['result'])
            except json.JSONDecodeError:
                return jsonify({"message": "Invalid JSON format in result field"}), 400
        
        # 如果未提供submission_time，设置为当前时间
        if not new_qol_data.get('submission_time'):
            new_qol_data['submission_time'] = db.func.current_timestamp()
        
        new_qol = db_operations.add_record(QoL, new_qol_data)
        if new_qol:
            return jsonify({"message": "QoL record added successfully!", "qol": new_qol.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding QoL record."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@crud_bp.route('/qols/<int:qol_id>', methods=['PUT'])
def edit_qol(qol_id):
    """Edit an existing QoL record."""
    existing_qol = db_operations.get_record_by_id(QoL, qol_id)
    if not existing_qol:
        return jsonify({"message": "QoL record not found."}), 404

    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form_data_from_request = request.json
    
    # 处理result字段：如果是字符串，尝试解析为JSON
    if 'result' in form_data_from_request and isinstance(form_data_from_request['result'], str):
        try:
            form_data_from_request['result'] = json.loads(form_data_from_request['result'])
        except json.JSONDecodeError:
            return jsonify({"message": "Invalid JSON format in result field"}), 400
    
    form = QoLForm(data=form_data_from_request, obj=existing_qol)

    if form.validate():
        updated_qol_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        updated_qol_data.pop('qol_id', None)  # Primary keys are usually not updated
        
        # 处理result字段
        if 'result' in updated_qol_data and isinstance(updated_qol_data['result'], str):
            try:
                updated_qol_data['result'] = json.loads(updated_qol_data['result'])
            except json.JSONDecodeError:
                return jsonify({"message": "Invalid JSON format in result field"}), 400
        
        updated_qol = db_operations.update_record(existing_qol, updated_qol_data)
        if updated_qol:
            return jsonify({"message": "QoL record updated successfully!", "qol": updated_qol.to_dict()})
        else:
            return jsonify({"message": "Error updating QoL record."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@crud_bp.route('/qols/<int:qol_id>', methods=['DELETE'])
def delete_qol(qol_id):
    """Delete a QoL record."""
    qol = db_operations.get_record_by_id(QoL, qol_id)
    if not qol:
        return jsonify({"message": "QoL record not found."}), 404

    if db_operations.delete_record(qol):
        return jsonify({"message": "QoL record deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting QoL record."}), 500
        
# --- CRUD Operations for Nurses ---
@crud_bp.route('/nurses', methods=['GET'])
def list_nurses():
    """List all nurses."""
    nurses = db_operations.get_all_records(Nurse)
    return jsonify([nurse.to_dict() for nurse in nurses])

@crud_bp.route('/nurses/<int:nurse_id>', methods=['GET'])
def get_nurse(nurse_id):
    """Get a single nurse by ID."""
    nurse = db_operations.get_record_by_id(Nurse, nurse_id)
    if not nurse:
        return jsonify({"message": "Nurse not found."}), 404
    return jsonify(nurse.to_dict())

@crud_bp.route('/nurses/search', methods=['GET'])
def search_nurses():
    """Search nurses by a specified field and value."""
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id'):
            field_value = int(field_value)
    except ValueError:
        pass

    nurses = db_operations.get_records_by_field(Nurse, field_name, field_value)
    if nurses:
        return jsonify([nurse.to_dict() for nurse in nurses])
    else:
        return jsonify({"message": f"No nurses found with {field_name} = {field_value}"}), 404

@crud_bp.route('/nurses', methods=['POST'])
def add_nurse():
    """Add a new nurse."""
    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form = NurseForm(data=request.json)
    if form.validate():
        nurse_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        nurse_data.pop('nurse_id', None)
        new_nurse = db_operations.add_record(Nurse, nurse_data)
        if new_nurse:
            return jsonify({"message": "Nurse added successfully!", "nurse": new_nurse.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding nurse."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@crud_bp.route('/nurses/<int:nurse_id>', methods=['PUT'])
def edit_nurse(nurse_id):
    """Edit an existing nurse."""
    nurse = db_operations.get_record_by_id(Nurse, nurse_id)
    if not nurse:
        return jsonify({"message": "Nurse not found."}), 404

    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form = NurseForm(data=request.json, obj=nurse)
    if form.validate():
        nurse_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        nurse_data.pop('nurse_id', None)
        updated_nurse = db_operations.update_record(nurse, nurse_data)
        if updated_nurse:
            return jsonify({"message": "Nurse updated successfully!", "nurse": updated_nurse.to_dict()})
        else:
            return jsonify({"message": "Error updating nurse."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@crud_bp.route('/nurses/<int:nurse_id>', methods=['DELETE'])
def delete_nurse(nurse_id):
    """Delete a nurse."""
    nurse = db_operations.get_record_by_id(Nurse, nurse_id)
    if not nurse:
        return jsonify({"message": "Nurse not found."}), 404

    if db_operations.delete_record(nurse):
        return jsonify({"message": "Nurse deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting nurse."}), 500

# --- CRUD Operations for Nurse Evaluations ---
@crud_bp.route('/nurse_evaluations', methods=['GET'])
def list_nurse_evaluations():
    """List all nurse evaluations."""
    evaluations = db_operations.get_all_records(NurseEvaluation)
    return jsonify([evaluation.to_dict() for evaluation in evaluations])

@crud_bp.route('/nurse_evaluations/<int:evaluation_id>', methods=['GET'])
def get_nurse_evaluation(evaluation_id):
    """Get a single nurse evaluation by ID."""
    evaluation = db_operations.get_record_by_id(NurseEvaluation, evaluation_id)
    if not evaluation:
        return jsonify({"message": "Nurse evaluation not found."}), 404
    return jsonify(evaluation.to_dict())

@crud_bp.route('/nurse_evaluations/search', methods=['GET'])
def search_nurse_evaluations():
    """Search nurse evaluations by a specified field and value."""
    field_name = request.args.get('field')
    field_value = request.args.get('value')

    if not field_name or not field_value:
        return jsonify({"message": "Missing 'field' or 'value' query parameters."}), 400

    try:
        if field_name.endswith('_id') or field_name == 'score':
            field_value = int(field_value)
    except ValueError:
        pass

    evaluations = db_operations.get_records_by_field(NurseEvaluation, field_name, field_value)
    if evaluations:
        return jsonify([evaluation.to_dict() for evaluation in evaluations])
    else:
        return jsonify({"message": f"No evaluations found with {field_name} = {field_value}"}), 404

@crud_bp.route('/nurse_evaluations', methods=['POST'])
def add_nurse_evaluation():
    """Add a new nurse evaluation."""
    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form = NurseEvaluationForm(data=request.json)
    if form.validate():
        evaluation_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        evaluation_data.pop('evaluation_id', None)
        new_evaluation = db_operations.add_record(NurseEvaluation, evaluation_data)
        if new_evaluation:
            return jsonify({"message": "Nurse evaluation added successfully!", "evaluation": new_evaluation.to_dict()}), 201
        else:
            return jsonify({"message": "Error adding evaluation."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@crud_bp.route('/nurse_evaluations/<int:evaluation_id>', methods=['PUT'])
def edit_nurse_evaluation(evaluation_id):
    """Edit an existing nurse evaluation."""
    evaluation = db_operations.get_record_by_id(NurseEvaluation, evaluation_id)
    if not evaluation:
        return jsonify({"message": "Nurse evaluation not found."}), 404

    if not request.is_json:
        return jsonify({"message": "Request must be JSON"}), 400

    form = NurseEvaluationForm(data=request.json, obj=evaluation)
    if form.validate():
        evaluation_data = {field.name: field.data for field in form if field.name != 'csrf_token'}
        evaluation_data.pop('evaluation_id', None)
        updated_evaluation = db_operations.update_record(evaluation, evaluation_data)
        if updated_evaluation:
            return jsonify({"message": "Nurse evaluation updated successfully!", "evaluation": updated_evaluation.to_dict()})
        else:
            return jsonify({"message": "Error updating evaluation."}), 500
    else:
        return jsonify({"message": "Validation failed", "errors": form.errors}), 400

@crud_bp.route('/nurse_evaluations/<int:evaluation_id>', methods=['DELETE'])
def delete_nurse_evaluation(evaluation_id):
    """Delete a nurse evaluation."""
    evaluation = db_operations.get_record_by_id(NurseEvaluation, evaluation_id)
    if not evaluation:
        return jsonify({"message": "Nurse evaluation not found."}), 404

    if db_operations.delete_record(evaluation):
        return jsonify({"message": "Nurse evaluation deleted successfully!"})
    else:
        return jsonify({"message": "Error deleting evaluation."}), 500

@crud_bp.route('/api/recovery_records/start', methods=['POST'])
def start_recovery_record():
    data = request.json
    if not data or 'user_id' not in data or 'plan_id' not in data:
        return jsonify({'error': 'user_id and plan_id are required'}), 400
    
    try:
        # 创建新的恢复记录
        new_record = RecoveryRecord(
            user_id=data['user_id'],
            # plan_id=data['plan_id'],
            record_date=datetime.now(),
            notes='in_progress'
        )
        db.session.add(new_record)
        db.session.commit()
        
        return jsonify({
            'record_id': new_record.record_id,
            'message': 'New recovery record created',
            'timestamp': datetime.now().isoformat()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@crud_bp.route('/api/user_recovery_plans/<int:user_id>', methods=['GET'])
def get_user_recovery_plan_by_user_id(user_id):
    """获取用户的康复计划信息"""
    try:
        # 查询用户当前的康复计划
        user_plan = db.session.query(
            UserRecoveryPlan.user_plan_id,
            UserRecoveryPlan.plan_id,
            RecoveryPlan.plan_name,
            RecoveryPlan.description
        ).join(
            RecoveryPlan, UserRecoveryPlan.plan_id == RecoveryPlan.plan_id
        ).filter(
            UserRecoveryPlan.user_id == user_id,
            UserRecoveryPlan.status == 'active'  # 假设只查询活跃的计划
        ).order_by(
            UserRecoveryPlan.assigned_date.desc()
        ).first()

        if not user_plan:
            return jsonify({"message": "No active recovery plan found for this user."}), 404

        # 将查询结果转换为字典
        plan_data = {
            "user_plan_id": user_plan.user_plan_id,
            "plan_id": user_plan.plan_id,
            "plan_name": user_plan.plan_name,
            "description": user_plan.description
        }

        return jsonify(plan_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@crud_bp.route('/users/check_followup_status', methods=['GET'])
def check_followup_status():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({"error": "缺少 user_id"}), 400

    user = db_operations.get_record_by_id(User, user_id)
    if not user or not user.registration_date:
        return jsonify({
            "is_followup_week": False,
            "has_completed_followup": True 
        }), 200

    try:
        today = date.today()
        registration_date = user.registration_date.date()
        
        days_since_registration = (today - registration_date).days
        
        current_week = days_since_registration // 7
        
        followup_weeks = [0, 2, 4, 6]
        
        is_followup_week = current_week in followup_weeks

        if not is_followup_week:
            return jsonify({
                "is_followup_week": False,
                "has_completed_followup": False 
            }), 200

        start_of_followup_week = registration_date + timedelta(days=current_week * 7)
        end_of_followup_week = start_of_followup_week + timedelta(days=6)

        completed_this_week = db.session.query(QoL).filter(
            and_(
                QoL.user_id == user_id,
                QoL.submission_time >= start_of_followup_week,
                QoL.submission_time < (end_of_followup_week + timedelta(days=1))
            )
        ).first()

        return jsonify({
            "is_followup_week": True,
            "has_completed_followup": completed_this_week is not None
        }), 200

    except Exception as e:
        print(f"Error in check_followup_status for user {user_id}: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

@crud_bp.route('/api/users/<int:user_id>/progress', methods=['GET'])
def get_user_progress(user_id):
    """
    计算用户从注册开始6周内的康复训练进度。
    """
    user = db_operations.get_record_by_id(User, user_id)
    if not user or not user.registration_date:
        return jsonify({
            "training_days": 0,
            "progress_percentage": 0
        }), 200

    try:
        registration_date = user.registration_date.date()
        end_date = registration_date + timedelta(days=42) # 6 weeks * 7 days
        today = date.today()

        total_days = 42
            
        # 查询在6周内的不重复的训练天数
        training_days_query = db.session.query(func.count(func.distinct(func.date(RecoveryRecord.record_date)))).filter(
            RecoveryRecord.user_id == user_id,
            RecoveryRecord.record_date >= registration_date,
            RecoveryRecord.record_date < (end_date + timedelta(days=1)) # 包含end_date
        )
        
        training_days = training_days_query.scalar() or 0
        
        progress_percentage = 0
        if total_days > 0:
            progress_percentage = round((training_days / total_days) * 100)
            
        return jsonify({
            "training_days": training_days,
            "progress_percentage": progress_percentage,
            "total_days_in_period": total_days
        }), 200

    except Exception as e:
        print(f"Error in get_user_progress for user {user_id}: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

@crud_bp.route('/user_recovery_plans/<int:user_id>/exercises', methods=['GET'])
def get_user_recovery_plan_exercises(user_id):
    """获取用户当前康复计划中包含的训练动作ID列表"""
    try:
        # 查询用户当前激活的康复计划
        user_plan = db.session.query(
            RecoveryPlan.description
        ).join(
            UserRecoveryPlan, RecoveryPlan.plan_id == UserRecoveryPlan.plan_id
        ).filter(
            UserRecoveryPlan.user_id == user_id,
            UserRecoveryPlan.status == 'active'
        ).order_by(
            UserRecoveryPlan.assigned_date.desc()
        ).first()

        if not user_plan or not user_plan.description:
            return jsonify({"error": "No active recovery plan with exercises found for this user."}), 404

        # 解析 description 字段以获取动作ID列表
        try:
            # The description is a JSON string like "[1,2,3]", so we use json.loads
            exercise_ids = json.loads(user_plan.description)
            if not isinstance(exercise_ids, list):
                raise ValueError("Description is not a valid JSON list.")
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            print(f"Error parsing exercise IDs from description for user {user_id}: {e}")
            return jsonify({"error": "Failed to parse exercise list from recovery plan."}), 500

        return jsonify({"exercise_ids": exercise_ids}), 200

    except Exception as e:
        print(f"Error in get_user_recovery_plan_exercises for user {user_id}: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

@crud_bp.route('/users/<int:user_id>/progress_summary', methods=['GET'])
def get_user_progress_summary(user_id):
    """获取用户健康记录页面的所有统计数据。"""
    try:
        # 1. 日历标记：锻炼过的日子和次数
        recovery_dates = db.session.query(
            func.date(RecoveryRecord.record_date).label('date'),
            func.count(RecoveryRecord.record_id).label('count')
        ).filter(
            RecoveryRecord.user_id == user_id
        ).group_by(func.date(RecoveryRecord.record_date)).all()

        markers = [{
            'year': r.date.year,
            'month': r.date.month,
            'day': r.date.day,
            'type': 'solar',
            'text': f'{r.count}次'
        } for r in recovery_dates]

        # 2. AI动作评估：最近一次锻炼记录的平均分
        latest_record = db.session.query(RecoveryRecord).filter(
            RecoveryRecord.user_id == user_id
        ).order_by(desc(RecoveryRecord.record_date)).first()

        ai_evaluation_score = '-'
        if latest_record:
            details = db.session.query(RecoveryRecordDetail).filter(
                RecoveryRecordDetail.record_id == latest_record.record_id,
                RecoveryRecordDetail.evaluation_details.isnot(None)
            ).all()

            scores = []
            for detail in details:
                try:
                    evaluation = json.loads(detail.evaluation_details)
                    if 'score' in evaluation:
                        scores.append(evaluation['score'])
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if scores:
                ai_evaluation_score = round(sum(scores) / len(scores), 1)


        # 3. 锻炼总天数
        total_training_days = len(recovery_dates)

        # 4. 生活质量得分
        latest_qol = db.session.query(QoL).filter(
            QoL.user_id == user_id
        ).order_by(desc(QoL.submission_time)).first()

        quality_of_life_score = '-'
        if latest_qol and latest_qol.result and 'scoring_result' in latest_qol.result:
            score_sum = 0
            for item in latest_qol.result['scoring_result']:
                if item.get('module_name') in [
                    "physical_wellbeing", "social_family_wellbeing", 
                    "emotional_wellbeing", "functional_wellbeing", "additional_concerns"
                ] and isinstance(item.get('value'), (int, float)):
                    score_sum += item.get('value')
            quality_of_life_score = round(score_sum,1)

        return jsonify({
            "markers": markers,
            "aiEvaluation": ai_evaluation_score,
            "totalTrainingDays": f"{total_training_days}天",
            "qualityOfLife": quality_of_life_score
        })

    except Exception as e:
        print(f"Error in get_user_progress_summary for user {user_id}: {e}")
        return jsonify({"error": "服务器内部错误"}), 500