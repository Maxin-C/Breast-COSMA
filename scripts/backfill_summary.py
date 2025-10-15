# filename: backfill_summaries.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import current_app
from api import create_app
from utils.database.models import db, ChatHistory
from utils.llm_service.consult import Consult

def backfill():
    app = create_app()
    with app.app_context():
        print("Starting backfill process for chat summaries...")
        
        records_to_update = db.session.query(ChatHistory).filter(ChatHistory.summary.is_(None)).all()
        
        if not records_to_update:
            print("No records need a summary. Exiting.")
            return

        print(f"Found {len(records_to_update)} records to summarize.")
        
        consult_service = Consult()
        
        for i, record in enumerate(records_to_update):
            print(f"Processing record {i+1}/{len(records_to_update)} with conversation_id: {record.conversation_id}...")
            
            # 确保对话历史不为空
            if not record.chat_history:
                print(f"  -> Skipping, chat history is empty.")
                continue

            try:
                # 生成摘要
                summary_text = consult_service.summarize_conversation(record.conversation_id)
                
                if summary_text:
                    record.summary = summary_text
                    print(f"  -> Summary generated successfully.")
                else:
                    print(f"  -> Failed to generate summary.")

            except Exception as e:
                print(f"  -> An error occurred for this record: {e}")

            # 为了防止长时间锁定，可以分批次提交
            if (i + 1) % 50 == 0:
                print("Committing batch to database...")
                db.session.commit()

        # 提交剩余的更改
        print("Committing final changes to database...")
        db.session.commit()
        print("Backfill process completed.")

if __name__ == "__main__":
    backfill()