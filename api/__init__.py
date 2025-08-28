from flask import Flask
from config import Config
from .extensions import db, compress, csrf
import os
from utils.chat.chat import ChatService

def create_app(config_class=Config):
    app = Flask(__name__, template_folder='../templates')
    app.config.from_object(config_class)

    # A more robust secret key setup
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_very_secret_key_for_development')
    app.config['WTF_CSRF_ENABLED'] = False

    db.init_app(app)
    compress.init_app(app)
    csrf.init_app(app)

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

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(media_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(crud_bp)

    return app