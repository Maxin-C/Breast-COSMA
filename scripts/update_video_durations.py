import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import create_app
from utils.database.models import db, Exercise

import cv2
import math

def get_video_duration_seconds(video_url: str) -> int:
    if not video_url:
        return 0
        
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"警告: 无法打开视频 URL '{video_url}'。")
        return 0
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps > 0 and frame_count > 0:
            duration_seconds = math.floor(frame_count / fps)
            return duration_seconds
        else:
            return 0
    except Exception as e:
        print(f"获取视频 '{video_url}' 时长时出错: {e}")
        return 0
    finally:
        cap.release()

def update_all_exercise_durations():
    app = create_app()
    with app.app_context():
        print("--- 开始批量更新训练视频时长 ---")

        # 1. 查询所有需要更新的练习
        # 条件：有 video_url 但 duration_seconds 为空(NULL)
        exercises_to_update = Exercise.query.filter(
            Exercise.video_url.isnot(None),
            Exercise.video_url != '',
            Exercise.duration_seconds.is_(None)
        ).all()

        if not exercises_to_update:
            print("数据库中所有视频时长均已更新，无需操作。")
            return

        print(f"发现 {len(exercises_to_update)} 个需要更新时长的训练视频。")
        updated_count = 0

        # 2. 遍历并更新每一条记录
        for i, exercise in enumerate(exercises_to_update, 1):
            print(f"\n[{i}/{len(exercises_to_update)}] 正在处理 Exercise ID: {exercise.exercise_id}...")
            print(f"  - 视频URL: {exercise.video_url}")

            try:
                # 3. 获取视频时长
                duration = get_video_duration_seconds(exercise.video_url)

                if duration > 0:
                    # 4. 更新数据库字段
                    exercise.duration_seconds = duration
                    print(f"  ✅ 成功获取时长: {duration} 秒，已更新记录。")
                    updated_count += 1
                else:
                    print(f"  ⚠️  无法获取视频时长，跳过此记录。")

            except Exception as e:
                print(f"  ❌ 处理 Exercise ID {exercise.exercise_id} 时发生错误: {e}")

        # 5. 提交所有更改到数据库
        if updated_count > 0:
            try:
                db.session.commit()
                print(f"\n--- 所有更改已成功提交到数据库！共更新了 {updated_count} 条记录。 ---")
            except Exception as e:
                db.session.rollback()
                print(f"\n--- 提交数据库时发生错误: {e} ---")
        else:
            print("\n--- 没有记录被更新。 ---")


if __name__ == '__main__':
    update_all_exercise_durations()