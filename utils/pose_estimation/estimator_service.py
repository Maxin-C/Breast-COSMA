from PIL import Image
from typing import List, Dict
import os

from .estimator import Estimator
from mmpose.apis import MMPoseInferencer

class ActionClassifierService:
    def __init__(self, pose_inferencer: MMPoseInferencer):
        estimator_config = {
            "device": os.getenv("INFERENCE_DEVICE", "cuda:0"),
            "clip_model_path": os.getenv("CLIP_MODEL_PATH"),
            "trained_model_path": os.getenv("ACTION_MODEL_PATH"),
            "label_info_path": os.getenv("LABEL_INFO_PATH"),
            "model_config_path": os.getenv("MODEL_CONFIG_PATH")
        }
        
        self.estimator = Estimator(estimator_config, pose_inferencer=pose_inferencer)

    def predict_frames_in_sprite(self, sprite_image: Image.Image, exercise_id: int) -> List[int]:
        if self.estimator is None:
            return []
        return self.estimator.predict(sprite_image, exercise_id)