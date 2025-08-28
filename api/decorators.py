from functools import wraps
from flask import session, request, jsonify, redirect, url_for

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'nurse_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for('auth.login_page')) # Note: url_for uses blueprint name
        return f(*args, **kwargs)
    return decorated_function