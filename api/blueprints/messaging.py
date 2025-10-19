from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta

from api.extensions import db
from api.decorators import login_required
from utils.database.models import User, ScheduledNotification
from utils.wechat_service.wechat import send_subscription_message

messaging_bp = Blueprint('messaging', __name__)

@messaging_bp.route('/send_notification', methods=['POST'])
@login_required
def send_notification():
    """
    一个示例接口，用于触发给某个用户发送打卡提醒。
    """
    req_data = request.json
    user_id_to_notify = req_data.get('user_id')
    if not user_id_to_notify:
        return jsonify({"error": "缺少 user_id"}), 400

    user = User.query.get(user_id_to_notify)
    if not user or not user.wechat_openid:
        return jsonify({"error": "用户不存在或没有 openid"}), 404

    template_id = 'GyMG05JSWzuZRImSH7N1x6kx6rd-GlEjyb_mGJZRRQg'
    message_data = {
        "thing1": { "value": "每日康复训练" },
        "thing4": { "value": "上肢康复训练" },
        "thing12": { "value": "浙江大学邵逸夫医院" }
    }
    
    result = send_subscription_message(user.wechat_openid, template_id, message_data, page="pages/home/home")
    
    if result["ok"]:
        return jsonify({"message": result["message"]}), 200
    else:
        return jsonify({"error": result["message"]}), 500

@messaging_bp.route('/schedule_notification', methods=['POST'])
def schedule_notification():
    """
    接收前端的用户授权，并安排第二天的通知。
    现在会接收 reminder_time 参数，并根据 user_id, template_id 和 scheduled_time 检查重复。
    """
    req_data = request.json
    user_id = req_data.get('user_id')
    template_id = req_data.get('template_id')
    reminder_time_str = req_data.get('reminder_time', '08:00') # 默认为 08:00

    if not user_id or not template_id:
        return jsonify({"error": "缺少 user_id 或 template_id"}), 400

    try:
        # 解析前端传来的时间
        reminder_hour, reminder_minute = map(int, reminder_time_str.split(':'))
        
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        # 使用用户设定的时间来创建第二天的提醒时间
        scheduled_time = tomorrow.replace(hour=reminder_hour, minute=reminder_minute, second=0, microsecond=0)

        # 检查是否已存在完全相同的待处理通知 (用户、模板、精确到分钟的时间)
        existing_notification = ScheduledNotification.query.filter(
            ScheduledNotification.user_id == user_id,
            ScheduledNotification.template_id == template_id,
            db.func.date(ScheduledNotification.scheduled_time) == scheduled_time.date(),
            db.func.hour(ScheduledNotification.scheduled_time) == scheduled_time.hour,
            db.func.minute(ScheduledNotification.scheduled_time) == scheduled_time.minute,
            ScheduledNotification.status == 'pending'
        ).first()

        if existing_notification:
            return jsonify({"message": "已存在待发送的通知，无需重复安排。"}), 200

        new_notification = ScheduledNotification(
            user_id=user_id,
            template_id=template_id,
            scheduled_time=scheduled_time
        )
        db.session.add(new_notification)
        db.session.commit()

        return jsonify({
            "message": f"订阅成功！我们将在明天 {reminder_time_str} 提醒您。",
            "scheduled_time": scheduled_time.isoformat()
        }), 201

    except Exception as e:
        db.session.rollback()
        print(f"Error scheduling notification: {e}")
        return jsonify({"error": "安排通知失败"}), 500

@messaging_bp.route('/check_subscription_status', methods=['GET'])
def check_subscription_status():
    """
    检查用户是否已有待处理的、针对某个模板的通知。
    """
    user_id = request.args.get('user_id', type=int)
    template_id = request.args.get('template_id')

    if not user_id or not template_id:
        return jsonify({"error": "缺少 user_id 或 template_id"}), 400

    try:
        # 查询数据库中是否存在状态为 'pending' 的匹配记录
        existing_notification = ScheduledNotification.query.filter_by(
            user_id=user_id,
            template_id=template_id,
            status='pending'
        ).first()

        if existing_notification:
            # 如果找到了，说明用户已订阅
            return jsonify({"isSubscribed": True}), 200
        else:
            # 如果没找到，说明用户未订阅
            return jsonify({"isSubscribed": False}), 200

    except Exception as e:
        print(f"检查订阅状态时出错: {e}")
        return jsonify({"error": "服务器内部错误"}), 500