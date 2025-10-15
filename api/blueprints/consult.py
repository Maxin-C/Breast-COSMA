from flask import Blueprint, request, jsonify, current_app
from api.extensions import db
from utils.database.models import ChatHistory

consult_bp = Blueprint('consult', __name__)

@consult_bp.route('/api/consult/messages', methods=['POST'])
def send_consult_message():
    """
    发送消息并获取回复。
    如果在请求体中提供了 conversation_id，则在现有对话中继续。
    如果未提供，则开始一个新的对话。
    """
    data = request.json
    if not data or 'message' not in data or 'user_id' not in data:
        return jsonify({'error': 'Fields "message" and "user_id" are required'}), 400

    try:
        # 从请求体中获取数据，conversation_id 是可选的
        user_id = data['user_id']
        message_text = data['message']
        conversation_id = data.get('conversation_id')  # 如果不存在则为 None

        # 调用新的 consult_service.chat 方法
        # 该方法内部处理了 conversation_id 为 None 的情况（即开始新对话）
        response = current_app.consult_service.chat(
            user_id=user_id,
            query=message_text,
            conversation_id=conversation_id
        )
        return jsonify(response), 200
    except Exception as e:
        # 记录异常可以帮助调试
        current_app.logger.error(f"Error in send_consult_message: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@consult_bp.route('/api/consult/conversations/<conversation_id>/messages', methods=['GET'])
def get_consult_messages(conversation_id):
    """获取指定对话的完整历史记录"""
    try:
        # 从 ChatHistory 模型中查询记录
        record = db.session.query(ChatHistory).filter(
            ChatHistory.conversation_id == conversation_id
        ).first()

        if not record:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # chat_history 字段直接存储了对话列表
        history = record.chat_history or []
        return jsonify(history), 200
    except Exception as e:
        current_app.logger.error(f"Error in get_consult_messages: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@consult_bp.route('/api/consult/conversations/<conversation_id>/summarize', methods=['GET'])
def get_conversation_summary(conversation_id):
    """获取指定对话的内容摘要"""
    try:
        # 调用 consult_service 中的新方法
        summary = current_app.consult_service.summarize_conversation(conversation_id)
        if not summary:
            return jsonify({'error': 'Conversation is empty or does not exist, cannot summarize.'}), 404
            
        return jsonify({'summary': summary}), 200
    except Exception as e:
        current_app.logger.error(f"Error in get_conversation_summary: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@consult_bp.route('/api/consult/user/context', methods=['GET'])
def get_user_consult_context():
    """获取用户上下文信息（用于调试）"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Query parameter "user_id" is required'}), 400
    
    try:
        # 调用 consult_service 中的内部方法获取上下文
        context = current_app.consult_service._get_user_context(int(user_id))
        return jsonify({'user_id': user_id, 'context': context})
    except ValueError:
        return jsonify({'error': 'user_id must be an integer'}), 400
    except Exception as e:
        current_app.logger.error(f"Error in get_user_consult_context: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500