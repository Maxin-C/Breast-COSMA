# batch_process_videos.py (Standalone Version)
import os
import time
from datetime import datetime
import cv2
from sqlalchemy import text

# --- Standalone Flask App Setup ---
# 1. 导入必要的 Flask 和配置组件
from flask import Flask
from config import Config

# 2. 导入在 models.py 中定义的 db 实例和需要的模型
#    注意：我们现在从 utils.database.models 导入 db，而不是从 api_test
from utils.database.models import db, RecoveryRecordDetail, VideoSliceImage

# 3. 创建一个最小化的 Flask 应用实例，其唯一目的是提供数据库上下文
app = Flask(__name__)
app.config.from_object(Config)

# 4. 将 SQLAlchemy 实例与我们的应用绑定
db.init_app(app)
# --- End of Setup ---


# --- Configuration ---
VIDEO_FOLDER = 'uploads/video'

# --- Helper Function ---
def _extract_frames_from_sprite(image_path):
    """从单个2x3雪碧图中提取6个视频帧"""
    try:
        sprite_image = cv2.imread(image_path)
        if sprite_image is None:
            print(f"    [Warning] Failed to read image: {image_path}")
            return []
        
        img_height, img_width, _ = sprite_image.shape
        frame_height = img_height // 2
        frame_width = img_width // 3
        
        frames = []
        for i in range(2): # 遍历行
            for j in range(3): # 遍历列
                frame = sprite_image[i*frame_height:(i+1)*frame_height, j*frame_width:(j+1)*frame_width]
                frames.append(frame)
        return frames
    except Exception as e:
        print(f"    [Error] Exception while extracting frames from {image_path}: {e}")
        return []

def process_unconverted_videos():
    """
    主处理函数：查找、验证、清理并转换所有未处理的视频记录。
    """
    print("--- Starting Batch Video Processing ---")
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER)
        print(f"Created video directory: {VIDEO_FOLDER}")

    # 1. 查找所有需要处理的 (record_id, exercise_id) 组合
    sql_query = text("""
        SELECT DISTINCT t1.record_id, t1.exercise_id
        FROM video_slice_images t1
        LEFT JOIN recovery_record_details t2 
        ON t1.record_id = t2.record_id AND t1.exercise_id = t2.exercise_id AND t2.video_path IS NOT NULL
        WHERE t2.record_detail_id IS NULL;
    """)
    unprocessed_tasks = db.session.execute(sql_query).fetchall()

    if not unprocessed_tasks:
        print("No new video records to process. Database is up to date.")
        return

    print(f"Found {len(unprocessed_tasks)} unprocessed record/exercise combinations to process.")

    # 2. 遍历每一个待处理的任务
    for task in unprocessed_tasks:
        record_id, exercise_id = task
        print(f"\n[Processing] Record ID: {record_id}, Exercise ID: {exercise_id}")

        # 3. 验证与清理
        image_slices_to_process = []
        all_paths_valid = True
        
        image_slices = VideoSliceImage.query.filter_by(
            record_id=record_id,
            exercise_id=exercise_id
        ).order_by(VideoSliceImage.timestamp, VideoSliceImage.slice_order).all()

        if not image_slices:
            print("    [Warning] No image slices found for this combination, skipping.")
            continue

        print(f"    Found {len(image_slices)} image slices. Validating paths...")
        for image_slice in image_slices:
            if os.path.exists(image_slice.image_path):
                image_slices_to_process.append(image_slice)
            else:
                print(f"    [Cleanup] Path not found: '{image_slice.image_path}'. Deleting record image_id: {image_slice.image_id}")
                db.session.delete(image_slice)
                all_paths_valid = False

        if not all_paths_valid:
            db.session.commit()
            print("    [Cleanup] Invalid records deleted.")

        if not image_slices_to_process:
            print("    [Warning] No valid image paths remain after cleanup, skipping video generation.")
            continue
            
        # 4. 生成视频
        print(f"    Extracting frames from {len(image_slices_to_process)} valid slices...")
        all_frames = []
        for image_slice in image_slices_to_process:
            frames = _extract_frames_from_sprite(image_slice.image_path)
            all_frames.extend(frames)

        if not all_frames:
            print("    [Error] Failed to extract any frames, skipping video generation.")
            continue
            
        print(f"    Total frames extracted: {len(all_frames)}. Creating video...")
        frame_height, frame_width, _ = all_frames[0].shape
        video_filename = f"record_{record_id}_exercise_{exercise_id}_{int(time.time())}.mp4"
        video_filepath = os.path.join(VIDEO_FOLDER, video_filename)
        
        fps = 6.0
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, (frame_width, frame_height))
        
        for frame in all_frames:
            video_writer.write(frame)
        video_writer.release()

        # 5. 创建新的 recovery_record_details 记录并保存视频路径
        new_detail = RecoveryRecordDetail(
            record_id=record_id,
            exercise_id=exercise_id,
            completion_timestamp=datetime.now(),
            video_path=video_filepath,
            evaluation_details="Video processed via batch script."
        )
        db.session.add(new_detail)
        db.session.commit()
        print(f"    [Success] Video created at '{video_filepath}'")
        print(f"    [Success] Created new recovery_record_details record with ID: {new_detail.record_detail_id}")

    print("\n--- Batch Video Processing Finished ---")

# --- Script Execution ---
if __name__ == '__main__':
    # 使用 with app.app_context() 来确保脚本可以访问数据库
    # 这里的 'app' 是我们在本文件中创建的最小化实例
    with app.app_context():
        process_unconverted_videos()