import os
import json
from mmpose.apis import MMPoseInferencer

from utils.detect_upper_body.detector import UpperBodyDetector
from utils.pose_estimation.estimator_service import ActionClassifierService
from utils.llm_service.consult import Consult
from utils.llm_service.report import ReportGenerator

mmpose_config_path = os.getenv("MMPOSE_CONFIG_PATH")
mmpose_cfg = json.load(open(mmpose_config_path, 'r'))

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

report_service = None

def init_dependent_services(db_session):
    global report_service
    if report_service is None:
        report_service = ReportGenerator(db_session=db_session)