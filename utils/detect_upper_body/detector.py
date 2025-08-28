import cv2
import numpy as np
import os
from types import SimpleNamespace
import json
from mmpose.apis import MMPoseInferencer

class UpperBodyDetector:
    def __init__(self, config_path, checkpoint_path, confidence_threshold=0.3, device="cuda:0", margin_ratio=0.1):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"错误: 配置文件 '{config_path}' 不存在。请检查路径或下载。")

        self.confidence_threshold = confidence_threshold
        self.margin_ratio = margin_ratio  # 边缘留白比例
        self.inferencer = None

        self._load_model(config_path, checkpoint_path, device)
        print("UpperBodyDetector 模型加载成功。")

    def _load_model(self, config_path, checkpoint_path, device="cuda:0"):
        """内部方法：加载MMPose模型。"""
        try:
            self.inferencer = MMPoseInferencer(config_path, checkpoint_path, device=device)
        except Exception as e:
            raise RuntimeError(f"初始化MMPose Inferencer失败: {e}\n请确保模型配置文件和权重文件路径正确，或者可以访问下载链接。")

    def detect(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                return False
        elif isinstance(image_path_or_array, np.ndarray):
            img = image_path_or_array
        else:
            raise TypeError("image_path_or_array 必须是图片路径字符串或numpy数组。")

        img_height, img_width, _ = img.shape
        margin_x = img_width * self.margin_ratio
        margin_y = img_height * self.margin_ratio

        result_generator = self.inferencer(img, return_vis=False, return_datasample=False)
        results = next(result_generator)
        
        # Access predictions safely, defaulting to an empty list
        predictions = results.get('predictions', []) 

        # Ensure predictions is not empty and structured as expected
        if not predictions or not isinstance(predictions, list) or not predictions[0]:
            return False

        # Access the list of person data within predictions[0]
        persons_in_image = predictions[0] if isinstance(predictions[0], list) else []
        
        if not persons_in_image:
            return False

        # Iterate through detected persons
        for person_data in persons_in_image:
            # Get keypoints (x, y coordinates) and keypoint_scores (confidence) separately
            keypoints = person_data.get('keypoints', [])
            keypoint_scores = person_data.get('keypoint_scores', [])
            
            # RTMPose models typically use COCO keypoint indexing:
            # 0: nose, 5: left_shoulder, 6: right_shoulder
            essential_upper_body_kpt_indices = [0, 5, 6, 7, 8] 

            detected_essential_kpts = []
            for idx in essential_upper_body_kpt_indices:
                if idx < len(keypoints) and \
                   idx < len(keypoint_scores) and \
                   keypoint_scores[idx] > self.confidence_threshold:
                    if isinstance(keypoints[idx], (list, np.ndarray)) and len(keypoints[idx]) >= 2:
                        detected_essential_kpts.append((keypoints[idx][0], keypoints[idx][1], idx)) # Store x, y coordinates and index
            
            if not detected_essential_kpts:
                continue 

            person_upper_body_in_frame = True
            shoulders_in_margin = True
            
            for kpt_x, kpt_y, idx in detected_essential_kpts:
                # 检查所有关键点是否在画面内
                if not (0 <= kpt_x <= img_width and 0 <= kpt_y <= img_height):
                    person_upper_body_in_frame = False
                    break
                
                # 特别检查肩部节点是否在边缘留白区域内
                if idx in [5, 6]:  # 5: left_shoulder, 6: right_shoulder
                    if (kpt_x < margin_x or kpt_x > img_width - margin_x or
                        kpt_y < margin_y or kpt_y > img_height - margin_y):
                        shoulders_in_margin = False
            
            if person_upper_body_in_frame and shoulders_in_margin:
                return True

        return False