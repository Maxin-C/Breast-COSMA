from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from api.extensions import db
# from utils.chat.chat import ChatService
from utils.database.models import MessageChat

chat_bp = Blueprint('chat', __name__)

# chat_service = ChatService()

@chat_bp.route('/api/chat/conversations', methods=['POST'])
def start_chat_conversation():
    data = request.json
    if not data or 'user_id' not in data:
        return jsonify({'error': 'user_id is required'}), 400
    
    try:
        conversation_id = current_app.chat_service.start_new_conversation(data['user_id'])
        return jsonify({
            'conversation_id': conversation_id,
            'message': 'New conversation started',
            'timestamp': datetime.now().isoformat()
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/chat/conversations/<conversation_id>/messages', methods=['POST'])
def send_chat_message(conversation_id):
    """发送消息并获取回复"""
    data = request.json
    if not data or 'message' not in data or 'user_id' not in data:
        return jsonify({'error': 'Both message and user_id are required'}), 400
    
    try:
        response = current_app.chat_service.process_user_message(
            user_id=data['user_id'],
            conversation_id=conversation_id,
            message_text=data['message']
        )
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/chat/conversations/<conversation_id>/messages', methods=['GET'])
def get_chat_messages(conversation_id):
    """获取对话历史"""
    try:
        messages = db.session.query(MessageChat).filter(
            MessageChat.conversation_id == conversation_id
        ).order_by(MessageChat.timestamp.asc()).all()
        
        return jsonify([message.to_dict() for message in messages])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/chat/user/context', methods=['GET'])
def get_user_chat_context():
    """获取用户上下文信息（用于调试）"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required as query parameter'}), 400
    
    try:
        context = current_app.chat_service._get_user_context(int(user_id))
        return jsonify({'context': context})
    except Exception as e:
        return jsonify({'error': str(e)}), 500