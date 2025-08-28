from flask_sqlalchemy import SQLAlchemy
from flask_compress import Compress
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
compress = Compress()
csrf = CSRFProtect()