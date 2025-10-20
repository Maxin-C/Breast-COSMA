from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date, time
from api.extensions import db
# db = SQLAlchemy()

# Base model with a to_dict method for JSON serialization
class Base(db.Model):
    __abstract__ = True

    def to_dict(self):
        # This method converts the SQLAlchemy model instance to a dictionary.
        # It handles datetime, date, and time objects for JSON serialization.
        data = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                data[column.name] = value.isoformat()
            elif isinstance(value, date):
                data[column.name] = value.isoformat()
            elif isinstance(value, time):
                data[column.name] = value.isoformat()
            else:
                data[column.name] = value
        return data

class User(Base):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    wechat_openid = db.Column(db.String(255))
    srrsh_id = db.Column(db.Integer)
    name = db.Column(db.String(100))
    phone_number = db.Column(db.String(20))
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)
    surgery_date = db.Column(db.DateTime)
    extubation_status = db.Column(db.String(100), nullable=False, default='未拔管')

    recovery_plans = db.relationship('UserRecoveryPlan', backref='user', lazy=True)
    calendar_schedules = db.relationship('CalendarSchedule', backref='user', lazy=True)
    recovery_records = db.relationship('RecoveryRecord', backref='user', lazy=True)

class RecoveryPlan(Base):
    __tablename__ = 'recovery_plans'
    plan_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    plan_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)

    user_recovery_plans = db.relationship('UserRecoveryPlan', backref='recovery_plan', lazy=True)

class Exercise(Base):
    __tablename__ = 'exercises'
    exercise_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    exercise_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    video_url = db.Column(db.String(255))
    image_url = db.Column(db.String(255))
    duration_seconds = db.Column(db.Integer)
    repetitions = db.Column(db.Integer)

    recovery_record_details = db.relationship('RecoveryRecordDetail', backref='exercise', lazy=True)
    video_slice_images = db.relationship('VideoSliceImage', backref='exercise', lazy=True)

class UserRecoveryPlan(Base):
    __tablename__ = 'user_recovery_plans'
    user_plan_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    plan_id = db.Column(db.Integer, db.ForeignKey('recovery_plans.plan_id'))
    assigned_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20))

class CalendarSchedule(Base):
    __tablename__ = 'calendar_schedule'
    schedule_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    schedule_date = db.Column(db.Date)
    schedule_time = db.Column(db.Time)
    type = db.Column(db.String(50))
    event_details = db.Column(db.Text)
    is_completed = db.Column(db.Boolean)
    completion_time = db.Column(db.DateTime)

class RecoveryRecord(Base):
    __tablename__ = 'recovery_records'
    record_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    record_date = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)
    evaluation_summary = db.Column(db.Text)

    record_details = db.relationship('RecoveryRecordDetail', backref='recovery_record', lazy=True)
    video_slice_images = db.relationship('VideoSliceImage', backref='recovery_record', lazy=True)

class RecoveryRecordDetail(Base):
    __tablename__ = 'recovery_record_details'
    record_detail_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    record_id = db.Column(db.Integer, db.ForeignKey('recovery_records.record_id'))
    exercise_id = db.Column(db.Integer, db.ForeignKey('exercises.exercise_id'))
    actual_duration_minutes = db.Column(db.Integer)
    actual_repetitions_completed = db.Column(db.Integer)
    evaluation_details = db.Column(db.Text)
    completion_timestamp = db.Column(db.DateTime)
    video_path = db.Column(db.String(255))

# class MessageChat(Base):
#     __tablename__ = 'messages_chat'
#     message_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     conversation_id = db.Column(db.String(255))
#     sender_id = db.Column(db.Integer)
#     sender_type = db.Column(db.Enum('user', 'assistant', 'professional'))
#     receiver_id = db.Column(db.Integer)
#     receiver_type = db.Column(db.Enum('user', 'assistant', 'professional'))
#     message_text = db.Column(db.Text)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)

#     def to_dict(self):
#         return {
#             'message_id': self.message_id,
#             'conversation_id': self.conversation_id,
#             'sender_id': self.sender_id,
#             'sender_type': self.sender_type,
#             'receiver_id': self.receiver_id,
#             'receiver_type': self.receiver_type,
#             'message_text': self.message_text,
#             'timestamp': self.timestamp.isoformat() if self.timestamp else None
#         }

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'

    message_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversation_id = db.Column(db.String(255), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    is_follow_up = db.Column(db.Boolean, default=False)
    chat_history = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    summary = db.Column(db.Text, nullable=True)

    user = db.relationship('User', backref=db.backref('chat_histories', lazy=True))

    def to_dict(self):
        return {
            'message_id': self.message_id,
            'conversation_id': self.conversation_id,
            'user_id': self.user_id,
            'is_follow_up': self.is_follow_up,
            'chat_history': self.chat_history,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'summary': self.summary
        }

    def __repr__(self):
        return f"<ChatHistory conversation_id='{self.conversation_id}' user_id={self.user_id}>"


class VideoSliceImage(Base):
    __tablename__ = 'video_slice_images'
    image_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey('exercises.exercise_id'))
    record_id = db.Column(db.Integer, db.ForeignKey('recovery_records.record_id'))
    slice_order = db.Column(db.Integer)
    image_path = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_part_of_action = db.Column(db.Boolean, default=False)

class QoL(Base):
    __tablename__ = 'qol_records'
    qol_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    form_name = db.Column(db.String(20), nullable=False)
    result = db.Column(db.JSON, nullable=False)  # JSON 类型字段
    submission_time = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    
    user = db.relationship('User', backref=db.backref('qol_records', lazy=True))

class Nurse(Base):
    __tablename__ = 'nurses'
    nurse_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False, unique=True)
    phone_number_suffix = db.Column(db.String(6), nullable=False)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)

    evaluations = db.relationship('NurseEvaluation', backref='nurse', lazy=True)

class NurseEvaluation(Base):
    __tablename__ = 'nurse_evaluations'
    evaluation_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    record_detail_id = db.Column(db.Integer, db.ForeignKey('recovery_record_details.record_detail_id'), nullable=False)
    nurse_id = db.Column(db.Integer, db.ForeignKey('nurses.nurse_id'), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    feedback_text = db.Column(db.Text)
    evaluation_timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Optional: Define relationship to RecoveryRecordDetail
    recovery_record_detail = db.relationship('RecoveryRecordDetail', backref='nurse_evaluations')

class ScheduledNotification(Base):
    __tablename__ = 'scheduled_notifications'
    notification_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    template_id = db.Column(db.String(255), nullable=False)
    scheduled_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.Enum('pending', 'sent', 'failed'), nullable=False, default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)