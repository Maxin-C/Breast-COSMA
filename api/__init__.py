from flask import Flask
from config import Config
from .extensions import db, compress, csrf, scheduler
import os
from utils.chat.chat import ChatService
from apscheduler.schedulers.background import BackgroundScheduler
from api.services.wechat import send_scheduled_notifications

def create_app(config_class=Config):
    app = Flask(__name__, template_folder='../templates')
    app.config.from_object(config_class)

    # A more robust secret key setup
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_very_secret_key_for_development')
    app.config['WTF_CSRF_ENABLED'] = False

    db.init_app(app)
    compress.init_app(app)
    csrf.init_app(app)

    if not scheduler.running:
        scheduler.configure(jobstores=app.config['SCHEDULER_JOBSTORES'])
        scheduler.add_job(
            func='api.services.wechat:scheduled_task',
            trigger='cron',
            hour=8,
            minute=0,
            second=5,
            id='send_notifications_cron_job',
            replace_existing=True
        )

        scheduler.add_job(
            # Use the STRING PATH to the importable function
            func='api.services.wechat:scheduled_task',
            trigger='interval',
            minutes=20,
            id='send_notifications_job',
            replace_existing=True
        )
        scheduler.start()
        print("Scheduler has been started.")

    with app.app_context():
        app.chat_service = ChatService()

        upload_folder = os.path.join(app.root_path, '..', 'uploads')
        video_folder = os.path.join(upload_folder, 'video')

        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
    
        db.create_all()

    # Import and register blueprints
    from .blueprints.auth import auth_bp
    from .blueprints.main import main_bp
    from .blueprints.media import media_bp
    from .blueprints.chat import chat_bp
    from .blueprints.crud import crud_bp
    from .blueprints.messaging import messaging_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(media_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(crud_bp)
    app.register_blueprint(messaging_bp)

    return app