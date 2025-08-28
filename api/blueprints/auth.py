from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for
from utils.database.models import Nurse
from api.extensions import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login')
def login_page():
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login_page'))

@auth_bp.route('/api/login', methods=['POST'])
def handle_login():
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "请输入用户名和密码"}), 400
    nurse = Nurse.query.filter_by(username=data['username'], phone_number_suffix=data['password']).first()
    if nurse:
        session['nurse_id'] = nurse.nurse_id
        session['nurse_name'] = nurse.name
        return jsonify({"message": "登录成功！"}), 200
    else:
        return jsonify({"error": "用户名或密码错误"}), 401