from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, DateTimeField, DateField, TextAreaField, BooleanField, SelectField, TimeField
from wtforms.validators import DataRequired, Optional, Length, NumberRange
from datetime import datetime, date, time

# Helper function to parse datetime strings safely
def parse_datetime_field(field_data):
    if field_data:
        try:
            return datetime.strptime(field_data, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass # Let WTForms validation handle invalid format if DataRequired is used
    return None

# Helper function to parse date strings safely
def parse_date_field(field_data):
    if field_data:
        try:
            return datetime.strptime(field_data, '%Y-%m-%d').date()
        except ValueError:
            pass
    return None

# Helper function to parse time strings safely
def parse_time_field(field_data):
    if field_data:
        try:
            return datetime.strptime(field_data, '%H:%M:%S').time()
        except ValueError:
            pass
    return None

class UserForm(FlaskForm):
    user_id = IntegerField('User ID', validators=[Optional()])
    wechat_openid = StringField('WeChat OpenID', validators=[Optional(), Length(max=255)])
    srrsh_id = IntegerField('SRRSH ID', validators=[Optional()])
    name = StringField('Name', validators=[DataRequired(), Length(max=100)])
    phone_number = StringField('Phone Number', validators=[Optional(), Length(max=20)])
    # For DateTimeField, ensure the format matches what you expect from input
    registration_date = DateTimeField('Registration Date (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])
    surgery_date = DateTimeField('Last Login Date (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])

class RecoveryPlanForm(FlaskForm):
    plan_id = IntegerField('Plan ID', validators=[Optional()])
    plan_name = StringField('Plan Name', validators=[DataRequired(), Length(max=100)])
    description = TextAreaField('Description', validators=[Optional()])
    start_date = DateField('Start Date (YYYY-MM-DD)', format='%Y-%m-%d', validators=[Optional()])
    end_date = DateField('End Date (YYYY-MM-DD)', format='%Y-%m-%d', validators=[Optional()])

class ExerciseForm(FlaskForm):
    exercise_id = IntegerField('Exercise ID', validators=[Optional()])
    exercise_name = StringField('Exercise Name', validators=[DataRequired(), Length(max=100)])
    description = TextAreaField('Description', validators=[Optional()])
    video_url = StringField('Video URL', validators=[Optional(), Length(max=255)])
    image_url = StringField('Image URL', validators=[Optional(), Length(max=255)])
    duration_seconds = IntegerField('Duration (Seconds)', validators=[Optional(), NumberRange(min=0)])
    repetitions = IntegerField('Repetitions', validators=[Optional(), NumberRange(min=0)])

class UserRecoveryPlanForm(FlaskForm):
    user_plan_id = IntegerField('User Plan ID', validators=[Optional()])
    user_id = IntegerField('User ID', validators=[DataRequired()])
    plan_id = IntegerField('Plan ID', validators=[DataRequired()])
    assigned_date = DateTimeField('Assigned Date (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])
    status = SelectField('Status', choices=[('active', 'Active'), ('completed', 'Completed'), ('cancelled', 'Cancelled')], validators=[DataRequired()])

class CalendarScheduleForm(FlaskForm):
    schedule_id = IntegerField('Schedule ID', validators=[Optional()])
    user_id = IntegerField('User ID', validators=[DataRequired()])
    schedule_date = DateField('Schedule Date (YYYY-MM-DD)', format='%Y-%m-%d', validators=[DataRequired()])
    schedule_time = TimeField('Schedule Time (HH:MM:SS)', format='%H:%M:%S', validators=[Optional()])
    type = StringField('Type', validators=[DataRequired(), Length(max=50)])
    event_details = TextAreaField('Event Details', validators=[Optional()])
    is_completed = BooleanField('Is Completed', validators=[Optional()])
    completion_time = DateTimeField('Completion Time (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])

class RecoveryRecordForm(FlaskForm):
    record_id = IntegerField('Record ID', validators=[Optional()])
    user_id = IntegerField('User ID', validators=[DataRequired()])
    record_date = DateTimeField('Record Date (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])
    notes = TextAreaField('Notes', validators=[Optional()])
    evaluation_summary = TextAreaField('Evaluation Summary', validators=[Optional()])

class RecoveryRecordDetailForm(FlaskForm):
    record_detail_id = IntegerField('Record Detail ID', validators=[Optional()]) # AUTO_INCREMENT, so optional for add
    record_id = IntegerField('Record ID', validators=[DataRequired()])
    exercise_id = IntegerField('Exercise ID', validators=[DataRequired()])
    actual_duration_minutes = IntegerField('Actual Duration (Minutes)', validators=[Optional(), NumberRange(min=0)])
    actual_repetitions_completed = IntegerField('Actual Repetitions Completed', validators=[Optional(), NumberRange(min=0)])
    evaluation_details = TextAreaField('AI Evaluation Result', validators=[Optional()])
    completion_timestamp = DateTimeField('Completion Timestamp (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])
    video_path = StringField('Video Path', validators=[Optional(), Length(max=255)])

class ChatHistoryForm(FlaskForm):
    message_id = IntegerField('Message ID', validators=[Optional()])
    conversation_id = StringField('Conversation ID', validators=[DataRequired(), Length(max=255)])
    user_id = IntegerField('User ID', validators=[DataRequired()])
    is_follow_up = BooleanField('Is Follow Up Conversation', validators=[Optional()])
    new_message_text = TextAreaField('New Message Text', validators=[DataRequired()])
    timestamp = DateTimeField('Timestamp', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])

class VideoSliceImageForm(FlaskForm):
    image_id = IntegerField('Image ID', validators=[Optional()])
    exercise_id = IntegerField('Exercise ID', validators=[DataRequired()])
    record_id = IntegerField('Record ID', validators=[DataRequired()])
    slice_order = IntegerField('Slice Order', validators=[Optional(), NumberRange(min=0)])
    image_path = StringField('Image Path', validators=[Optional(), Length(max=255)])
    timestamp = DateTimeField('Timestamp (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])
    is_part_of_action = BooleanField('Is Part of Action', validators=[Optional()])

class QoLForm(FlaskForm):
    qol_id = IntegerField('qol ID', validators=[Optional()])
    form_name = StringField('Form Name', validators=[DataRequired()])
    user_id = IntegerField('User Id', validators=[DataRequired()])
    result = StringField('Result JSON', validators=[DataRequired()])
    submission_time = StringField('Submission Time', validators=[Optional()])

class NurseForm(FlaskForm):
    nurse_id = IntegerField('Nurse ID', validators=[Optional()])
    name = StringField('Name', validators=[DataRequired(), Length(max=100)])
    username = StringField('Username', validators=[DataRequired(), Length(max=100)])
    phone_number_suffix = StringField('Phone Number Suffix', validators=[DataRequired(), Length(min=6, max=6)])
    registration_date = DateTimeField('Registration Date', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])

class NurseEvaluationForm(FlaskForm):
    evaluation_id = IntegerField('Evaluation ID', validators=[Optional()])
    record_detail_id = IntegerField('Record Detail ID', validators=[DataRequired()])
    nurse_id = IntegerField('Nurse ID', validators=[DataRequired()])
    score = IntegerField('Score', validators=[DataRequired(), NumberRange(min=0, max=9)])
    feedback_text = TextAreaField('Feedback Text', validators=[Optional()])
    evaluation_timestamp = DateTimeField('Timestamp', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])

class ScheduledNotificationForm(FlaskForm):
    notification_id = IntegerField('Notification ID', validators=[Optional()])
    user_id = IntegerField('User ID', validators=[DataRequired()])
    template_id = StringField('Template ID', validators=[DataRequired(), Length(max=255)])
    scheduled_time = DateTimeField('Scheduled Time (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[DataRequired()])
    status = SelectField('Status', choices=[('pending', 'Pending'), ('sent', 'Sent'), ('failed', 'Failed')], validators=[DataRequired()])
    created_at = DateTimeField('Created At (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])