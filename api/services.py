import os
import json
import threading
import queue
import time
from PIL import Image
from flask import current_app
from mmpose.apis import MMPoseInferencer

from utils.detect_upper_body.detector import UpperBodyDetector
from utils.pose_estimation.estimator_service import ActionClassifierService
from utils.llm_service.consult import Consult
from utils.llm_service.report import ReportGenerator
from utils.database.models import db, VideoSliceImage

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

mmpose_config_path = os.getenv("MMPOSE_CONFIG_PATH")
with open(mmpose_config_path, 'r') as f:
    mmpose_cfg = json.load(f)

shared_pose_inferencer = MMPoseInferencer(
    pose2d=mmpose_cfg['pose2d_config'],
    pose2d_weights=mmpose_cfg['pose2d_checkpoint'],
    pose3d=mmpose_cfg['pose3d_config'],
    pose3d_weights=mmpose_cfg['pose3d_checkpoint'],
    det_model=mmpose_cfg['det_config'],
    det_weights=mmpose_cfg['det_checkpoint'],
    device=os.getenv("INFERENCE_DEVICE", "cuda:0")
)

upper_body_detector = UpperBodyDetector(
    config_path=mmpose_cfg['pose2d_config'],
    checkpoint_path=mmpose_cfg['pose2d_checkpoint'],
    device=os.getenv("INFERENCE_DEVICE", "cuda:0")
)

action_classifier_service = ActionClassifierService(
    pose_inferencer=shared_pose_inferencer
)

consult_service = Consult()

# ReportService 依赖 db_session，仍在 init_dependent_services 中初始化
report_service = None

# --- 3. 新增：推理任务队列 ---
inference_queue = queue.Queue()

def _inference_worker(app):
    """后台工作线程：从队列获取图片并执行推理，避免阻塞 API"""
    print("Background inference worker started.")
    # 需要手动推入应用上下文，因为线程中没有请求上下文
    with app.app_context():
        while True:
            try:
                task = inference_queue.get()
                if task is None: break 
                
                slice_id, image_path, exercise_id = task
                
                try:
                    pil_image = Image.open(image_path)
                    
                    frame_results = action_classifier_service.predict_frames_in_sprite(
                        pil_image, int(exercise_id)
                    )
                    
                    is_part_of_action = all(x == 1 for x in frame_results)
                    
                    slice_record = VideoSliceImage.query.get(slice_id)
                    if slice_record:
                        slice_record.is_part_of_action = is_part_of_action
                        db.session.commit()
                        print(f"[Worker] Slice {slice_id} processed. Action: {is_part_of_action}")
                    else:
                        print(f"[Worker Warning] Slice {slice_id} not found in DB.")

                except Exception as e:
                    db.session.rollback()
                    print(f"[Worker Error] Processing failed for {image_path}: {e}")
                finally:
                    inference_queue.task_done()
                    
            except Exception as e:
                print(f"[Worker Critical Error] {e}")

def init_dependent_services(db_session):
    torch.set_num_threads(2)
    global report_service
    if report_service is None:
        report_service = ReportGenerator(db_session=db_session)
    
    try:
        app = current_app._get_current_object()
        t = threading.Thread(target=_inference_worker, args=(app,), daemon=True)
        t.start()
        print("Inference worker thread launched.")
    except Exception as e:
        print(f"Failed to start inference worker: {e}")