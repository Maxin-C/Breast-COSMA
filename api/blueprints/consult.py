from flask import Blueprint, request, jsonify, current_app
from api.extensions import db
from utils.database.models import ChatHistory

consult_bp = Blueprint('consult', __name__)

@consult_bp.route('/consult/messages', methods=['POST'])
def send_consult_message():
    data = request.json
    if not data or 'message' not in data or 'user_id' not in data:
        return jsonify({'error': 'Fields "message" and "user_id" are required'}), 400

    try:
        user_id = data['user_id']
        message_text = data['message']
        conversation_id = data.get('conversation_id') 

        mode = data.get('mode', 'consult').lower()
        end_conversation = data.get('end_conversation', False)

        
        if mode not in ['consult', 'followup']:
            return jsonify({'error': 'Invalid mode specified. Must be "consult" or "followup".'}), 400

        response = current_app.consult_service.process_message(
            user_id=user_id,
            query=message_text,
            conversation_id=conversation_id,
            mode=mode,
            end_conversation=end_conversation
        )
        return jsonify(response), 200
    except Exception as e:
        current_app.logger.error(f"Error in send_consult_message: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@consult_bp.route('/consult/conversations/<conversation_id>/messages', methods=['GET'])
def get_consult_messages(conversation_id):
    try:
        record = db.session.query(ChatHistory).filter(
            ChatHistory.conversation_id == conversation_id
        ).first()

        if not record:
            return jsonify({'error': 'Conversation not found'}), 404
        
        history = record.chat_history or []
        return jsonify(history), 200
    except Exception as e:
        current_app.logger.error(f"Error in get_consult_messages: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@consult_bp.route('/consult/conversations/<conversation_id>/summarize', methods=['GET'])
def get_conversation_summary(conversation_id):
    try:
        summary = current_app.consult_service.summarize_conversation(conversation_id)
        if not summary:
            return jsonify({'error': 'Conversation is empty or does not exist, cannot summarize.'}), 404
            
        return jsonify({'summary': summary}), 200
    except Exception as e:
        current_app.logger.error(f"Error in get_conversation_summary: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@consult_bp.route('/consult/user/context', methods=['GET'])
def get_user_consult_context():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Query parameter "user_id" is required'}), 400
    
    try:
        context = current_app.consult_service._get_user_context(int(user_id))
        return jsonify({'user_id': user_id, 'context': context})
    except ValueError:
        return jsonify({'error': 'user_id must be an integer'}), 400
    except Exception as e:
        current_app.logger.error(f"Error in get_user_consult_context: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500