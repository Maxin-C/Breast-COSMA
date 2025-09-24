import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://flask_user:password@localhost/breast_cosma_db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    WECHAT_APPID = os.environ.get('WECHAT_APPID')
    WECHAT_APPSECRET = os.environ.get('WECHAT_APPSECRET')

    SCHEDULER_JOBSTORES = {
        'default': {
            'type': 'sqlalchemy',
            'url': os.environ.get('DATABASE_URL', 'sqlite:///your_database.db')
        }
    }
    SCHEDULER_API_ENABLED = True