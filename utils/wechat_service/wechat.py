import requests
import time
from datetime import datetime
from flask import current_app

from api.extensions import db
from utils.database.models import User, ScheduledNotification

wechat_access_token_cache = {
    "access_token": "",
    "expires_at": 0
}

def get_wechat_access_token():
    """
    获取并缓存微信小程序的 access_token。
    """
    global wechat_access_token_cache
    now = int(time.time())

    if wechat_access_token_cache["access_token"] and wechat_access_token_cache["expires_at"] > now + 300:
        return wechat_access_token_cache["access_token"]

    appid = current_app.config.get('WECHAT_APPID')
    appsecret = current_app.config.get('WECHAT_APPSECRET')
    if not appid or not appsecret:
        print("Error: WECHAT_APPID 和 WECHAT_APPSECRET 未配置。")
        return None

    url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={appid}&secret={appsecret}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "access_token" in data:
            wechat_access_token_cache["access_token"] = data["access_token"]
            wechat_access_token_cache["expires_at"] = now + data["expires_in"]
            return data["access_token"]
        else:
            print(f"Error getting access_token: {data}")
            return None
    except requests.RequestException as e:
        print(f"Request to get access_token failed: {e}")
        return None

def send_subscription_message(openid, template_id, data, page=None):
    """
    封装的发送订阅消息的通用函数。
    """
    access_token = get_wechat_access_token()
    if not access_token:
        return {"ok": False, "message": "无法获取 access_token"}

    url = f"https://api.weixin.qq.com/cgi-bin/message/subscribe/send?access_token={access_token}"
    
    payload = { "touser": openid, "template_id": template_id, "data": data }
    if page:
        payload["page"] = page

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get("errcode") == 0:
            print(f"成功发送订阅消息给 {openid}")
            return {"ok": True, "message": "发送成功"}
        else:
            print(f"发送订阅消息失败: {result}")
            return {"ok": False, "message": f"发送失败: {result.get('errmsg')}"}
    except requests.RequestException as e:
        print(f"Request to send subscription message failed: {e}")
        return {"ok": False, "message": f"请求失败: {e}"}

def send_scheduled_notifications():
    """
    由调度器定时执行的函数，用于查询并发送所有到期的通知。
    """
    print(f"[{datetime.now()}] 任务执行：检查待发送的订阅消息...")
    now = datetime.now()
    
    notifications_to_send = ScheduledNotification.query.filter(
        ScheduledNotification.status == 'pending',
        ScheduledNotification.scheduled_time <= now
    ).all()

    if not notifications_to_send:
        print("没有需要发送的消息。")
        return

    for notification in notifications_to_send:
        user = User.query.get(notification.user_id)
        if not user or not user.wechat_openid:
            notification.status = 'failed'
            print(f"失败：找不到用户 {notification.user_id} 或其 openid。")
            continue

        message_data = {
            "thing1": { "value": "每日康复训练" },
            "thing4": { "value": "请开始今天的训练吧！" },
            "thing12": { "value": "乳腺癌术后智能康复助手" }
        }
        
        result = send_subscription_message(
            user.wechat_openid,
            notification.template_id,
            message_data,
            page="pages/home/home"
        )
        notification.status = 'sent' if result["ok"] else 'failed'
    
    db.session.commit()
    print(f"任务完成：处理了 {len(notifications_to_send)} 条通知。")

def scheduled_task():
    from api import create_app

    app = create_app()
    with app.app_context():
        # This calls the actual logic function, which is defined in this same file
        send_scheduled_notifications()