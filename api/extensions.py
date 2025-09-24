from flask_sqlalchemy import SQLAlchemy
from flask_compress import Compress
from flask_wtf.csrf import CSRFProtect
from apscheduler.schedulers.background import BackgroundScheduler

db = SQLAlchemy()
compress = Compress()
csrf = CSRFProtect()
scheduler = BackgroundScheduler()