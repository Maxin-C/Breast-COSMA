# batch_processor.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import current_app
from api import create_app
from utils.database.models import db
from utils.database.models import RecoveryRecord, VideoSliceImage

def run_batch_processing():
    """
    对数据库中所有康复记录进行批量评估和总结。
    """
    app = create_app()
    with app.app_context():
        print("--- 开始批量处理任务 ---")
        
        # 获取 report_service 服务实例
        report_service = current_app.report_service
        
        # 1. 查询所有需要处理的主记录
        all_records = RecoveryRecord.query.all()
        if not all_records:
            print("数据库中没有找到康复记录，任务结束。")
            return
            
        total_records = len(all_records)
        print(f"共找到 {total_records} 条康复记录需要处理。")

        # 2. 遍历每一条主记录
        for i, record in enumerate(all_records, 1):
            print(f"\n[{i}/{total_records}] 正在处理 Record ID: {record.record_id}...")
            
            try:
                # 3. 查找该记录下所有不重复的 exercise_id
                #    这代表了该次康复记录中包含的所有训练项目
                exercise_ids_tuples = db.session.query(VideoSliceImage.exercise_id)\
                    .filter_by(record_id=record.record_id)\
                    .distinct()\
                    .all()
                
                if not exercise_ids_tuples:
                    print(f"  - Record ID: {record.record_id} 没有找到关联的训练图片(VideoSliceImage)，跳过。")
                    continue
                
                exercise_ids = [item[0] for item in exercise_ids_tuples]
                print(f"  - 发现 {len(exercise_ids)} 个训练项目: {exercise_ids}")

                # 4. 对每个训练项目进行评估
                for ex_id in exercise_ids:
                    print(f"    -> 正在评估 Exercise ID: {ex_id}...")
                    try:
                        eval_result = report_service.evaluate_and_save(record.record_id, ex_id)
                        if eval_result.get('success'):
                            print(f"    ✅ 评估成功。Detail ID: {eval_result.get('detail_id')}")
                        else:
                            print(f"    ❌ 评估失败: {eval_result.get('error')}")
                    except Exception as e:
                        print(f"    💥 评估 Exercise ID {ex_id} 时发生严重错误: {e}")

                # 5. 在所有项目评估完成后，生成并保存总结报告
                print(f"  - 正在为 Record ID: {record.record_id} 生成总结报告...")
                try:
                    summary_result = report_service.summarize_and_save_for_record(record.record_id)
                    if summary_result.get('success'):
                        print(f"  ✅ 总结报告生成并保存成功。")
                    else:
                        print(f"  ❌ 总结失败: {summary_result.get('error')}")
                except Exception as e:
                    print(f"  💥 生成总结报告时发生严重错误: {e}")

            except Exception as e:
                print(f"处理 Record ID {record.record_id} 时发生顶级异常，跳过此记录: {e}")

        print("\n--- 所有任务处理完毕 ---")

if __name__ == '__main__':
    run_batch_processing()