# api/__init__.py

from flask import Flask
from config import Config
import json
from .extensions import db, compress, csrf, scheduler
import os

from . import services
from utils.wechat_service.wechat import send_scheduled_notifications
from apscheduler.schedulers.background import BackgroundScheduler

def create_app(config_class=Config):
    app = Flask(__name__, template_folder='../templates')
    app.config.from_object(config_class)

    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_very_secret_key_for_development')
    app.config['WTF_CSRF_ENABLED'] = False

    db.init_app(app)
    compress.init_app(app)
    csrf.init_app(app)

    if not scheduler.running:
        scheduler.configure(jobstores=app.config['SCHEDULER_JOBSTORES'])
        scheduler.add_job(
            func='utils.wechat_service.wechat:scheduled_task',
            trigger='cron',
            hour=8,
            minute=0,
            second=5,
            id='send_notifications_cron_job',
            replace_existing=True
        )
        scheduler.add_job(
            func='utils.wechat_service.wechat:scheduled_task',
            trigger='interval',
            minutes=20,
            id='send_notifications_job',
            replace_existing=True
        )
        scheduler.start()
        print("Scheduler has been started.")

    with app.app_context():
        services.init_dependent_services(db.session)

        app.pose_inferencer = services.shared_pose_inferencer
        app.upper_body_detector = services.upper_body_detector
        app.action_classifier_service = services.action_classifier_service
        app.consult_service = services.consult_service
        app.report_service = services.report_service 

        upload_folder = os.getenv("SLICE_SAVE_PATH", "uploads/slices")
        video_folder = os.getenv("VIDEO_SAVE_PATH", "uploads/videos")

        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
    
        db.create_all()

    from .blueprints.auth import auth_bp
    from .blueprints.main import main_bp
    from .blueprints.report import report_bp
    from .blueprints.consult import consult_bp
    from .blueprints.crud import crud_bp
    from .blueprints.messaging import messaging_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(consult_bp)
    app.register_blueprint(crud_bp)
    app.register_blueprint(messaging_bp)

    return app