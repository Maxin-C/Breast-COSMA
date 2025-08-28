import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://flask_user:password@localhost/breast_cosma_db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False