import cv2
import numpy as np
import os
from types import SimpleNamespace
import json
from mmpose.apis import MMPoseInferencer

class UpperBodyDetector:
    def __init__(self, config_path, checkpoint_path, confidence_threshold=0.3, device="cuda:0"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"错误: 配置文件 '{config_path}' 不存在。请检查路径或下载。")

        self.confidence_threshold = confidence_threshold
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
        flag = True
        for person_data in persons_in_image:
            # Get keypoints (x, y coordinates) and keypoint_scores (confidence) separately
            keypoints = person_data.get('keypoints', [])
            keypoint_scores = person_data.get('keypoint_scores', [])
            
            # RTMPose models typically use COCO keypoint indexing:
            # 0: nose, 5: left_shoulder, 6: right_shoulder
            essential_upper_body_kpt_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12] 
            person_upper_body_in_frame = True
            print(keypoint_scores)
            for idx in essential_upper_body_kpt_indices:
                if idx < len(keypoints) and idx < len(keypoint_scores):
                    if keypoint_scores[idx] < self.confidence_threshold:
                        person_upper_body_in_frame = False
                        break
            flag = flag and person_upper_body_in_frame
        return flag

            # detected_essential_kpts = []
            # for idx in essential_upper_body_kpt_indices:
            #     if idx < len(keypoints) and \
            #        idx < len(keypoint_scores) and \
            #        keypoint_scores[idx] > self.confidence_threshold:
            #         if isinstance(keypoints[idx], (list, np.ndarray)) and len(keypoints[idx]) >= 2:
            #             detected_essential_kpts.append(keypoints[idx]) # Store x, y coordinates
            # if not detected_essential_kpts:
            #     continue 

            # # Assume this person's upper body is in frame until proven otherwise
            # person_upper_body_in_frame = True
            # for kpt_x, kpt_y in detected_essential_kpts:
            #     # Direct check: Is keypoint within image bounds (0 to width/height)?
            #     if not (0 <= kpt_x <= img_width and 0 <= kpt_y <= img_height):
            #         person_upper_body_in_frame = False
            #         break # As soon as one keypoint is out, this person is out
            
            # # If all essential keypoints for this person are in frame, we're done
            # if person_upper_body_in_frame:
            #     return True 

        # If loop completes and no person's upper body was entirely in frame
        return False

# --- 示例使用 (不用于Flask) ---
if __name__ == '__main__':

    mmpose_config_path="utils/pose_estimation/mmpose_config.json"
    mmpose_config = SimpleNamespace(**json.load(open(mmpose_config_path,'r')))

    device = "cuda:0" # Or "cpu" if you don't have a GPU

    detector = UpperBodyDetector(mmpose_config.pose2d_config, mmpose_config.pose2d_checkpoint, device=device)
    
    _test_image_path = 'sprite_sheet_pil.jpg' 

    is_in_frame = detector.detect(_test_image_path)
    print(f"\n测试结果 (文件路径): 上半身在画面内: {is_in_frame}")
