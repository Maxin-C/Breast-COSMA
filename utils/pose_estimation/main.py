import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from mmpose.apis import MMPoseInferencer
from types import SimpleNamespace
import torch
from PIL import Image
import cv2

from utils.pose_estimation.model import PoseEstimationModel

class FrameDataset():
    def __init__(self, image_path, image_info, clip_processor, sk_inferencer):
        self.image_path = image_path
        self.sk_inferencer = sk_inferencer

        self.frames = self._get_frames(
            cols=image_info['cols'], 
            rows=image_info['rows'], 
            total_frames=image_info['total_frames'])

        self.clip_tensor = clip_processor(images=self.frames, return_tensors="pt").pixel_values.unsqueeze(dim=0)
        self.sk_tensor = torch.tensor(self._get_sk_tensor(), dtype=torch.float32).unsqueeze(dim=0)

    def _get_frames(self, cols, rows, total_frames):
        img = Image.open(self.image_path)
        single_frame_width = img.width // cols
        single_frame_height = img.height // rows
        frames = []
        frames_count = 0

        for r in range(rows):
            for c in range(cols):
                if frames_count >= total_frames:
                    break
                
                left = c * single_frame_width
                top = r * single_frame_height
                right = left + single_frame_width
                bottom = top + single_frame_height

                single_frame_img = img.crop((left, top, right, bottom))
                frames.append(single_frame_img)
                frames_count += 1
        return frames
    
    def _get_sk_tensor(self):
        frames_bgr = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in self.frames]

        result_generator = self.sk_inferencer(frames_bgr, show=False, save_results=False)
        all_keypoints = [np.array(result['predictions'][0][0]['keypoints']) for result in result_generator]
        keypoints_array = np.stack(all_keypoints)
        angles = self._calculate_upper_body_angles_batch(keypoints_array)
        upper_body = keypoints_array[:, :13, :].reshape(keypoints_array.shape[0], -1)
        return np.concatenate((upper_body, angles), axis=1)

    def _calculate_angle_batch(self, a, b, c):
        ba = a - b 
        bc = c - b
        
        dot_product = np.einsum('ij,ij->i', ba, bc)
        norm_ba = np.linalg.norm(ba, axis=1)
        norm_bc = np.linalg.norm(bc, axis=1)
        
        cosine_angle = dot_product / (norm_ba * norm_bc + 1e-7)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        return np.degrees(np.arccos(cosine_angle))

    def _calculate_upper_body_angles_batch(self, keypoints):
        if keypoints.shape[1:] != (17, 3):
            raise ValueError("输入关键点形状应为 (n_frames, 17, 3)")
        
        left_shoulder = keypoints[:, 5]
        right_shoulder = keypoints[:, 6]
        left_elbow = keypoints[:, 7]
        right_elbow = keypoints[:, 8]
        left_wrist = keypoints[:, 9]
        right_wrist = keypoints[:, 10]
        left_hip = keypoints[:, 11]
        right_hip = keypoints[:, 12]
        
        left_elbow_angle = self._calculate_angle_batch(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self._calculate_angle_batch(right_shoulder, right_elbow, right_wrist)
        left_shoulder_angle = self._calculate_angle_batch(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = self._calculate_angle_batch(right_elbow, right_shoulder, right_hip)
        
        return np.stack([
            left_elbow_angle,
            right_elbow_angle,
            left_shoulder_angle,
            right_shoulder_angle
        ], axis=1)

    def get_model_input(self):
        return self.clip_tensor, self.sk_tensor

class PoseEstimation():
    def __init__(
        self, 
        clip_model_path="/mnt/pvc-data.common/ChenZikang/huggingface/openai/clip-vit-large-patch14", 
        model_dict_path="utils/pose_estimation/model_dict/model_20250609_115936_acc98.48.pth",
        model_config_path="utils/pose_estimation/model_config.json",
        mmpose_config_path="utils/pose_estimation/mmpose_config.json",
        device="cuda:0"):
        self.device = device

        clip_model = CLIPModel.from_pretrained(clip_model_path)
        model_config = json.load(open(model_config_path, 'r'))
        self.model = PoseEstimationModel(
            clip_model=clip_model, 
            model_config=SimpleNamespace(**model_config)
        )
        try:
            self.model.load_state_dict(torch.load(model_dict_path))
        except Exception as e:
            print(e)
            print("Model dict loading failed.")
        self.model.to(device)

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        mmpose_config = SimpleNamespace(**json.load(open(mmpose_config_path,'r')))
        self.sk_inferencer = MMPoseInferencer(
            pose2d=mmpose_config.pose2d_config,
            pose2d_weights=mmpose_config.pose2d_checkpoint,
            det_model=mmpose_config.det_config,
            det_weights=mmpose_config.det_checkpoint,
            pose3d=mmpose_config.pose3d_config,
            pose3d_weights=mmpose_config.pose3d_checkpoint,
            device=device
        )

        action_info = ['Hold Loose Fist', 'Rotate Wrist', 'Flex and Extend Forearm', 'Touch Shoulder', 'Touch Ear', 'Deep Breathing Exercise', 'Hold Elbow and Comb Hair Exercise', 'Shrug Exercise', 'Torso Twist Exercise', 'Reach Overhead and Touch Ear', 'Pendulum Exercise', 'Wall Climbing Exercise', 'Circle Drawing Exercise', 'Pulley Exercise', 'Wash Back Exercise']
        self.action_info_zh = ['握松拳', '旋转手腕', '伸曲前臂', '摸肩膀', '摸耳朵', '深呼吸', '梳头运动', '耸肩运动', '转体运动', '过顶触耳', '钟摆运动', '爬墙运动', '画圈运动', '滑轮运动', '洗后背运动']
        actions = [f"human action of {i}" for i in action_info]
        self.action_inputs = self.clip_processor(text=actions, return_tensors="pt", padding=True, truncation=True).to(device)

    def inference(self, image_path, image_info):
        dataset = FrameDataset(image_path, image_info, self.clip_processor, self.sk_inferencer)
        clip_tensor, sk_tensor = dataset.get_model_input()
        clip_tensor, sk_tensor = clip_tensor.to(self.device), sk_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(clip_tensor, sk_tensor, self.action_inputs)
            _, predicted = torch.max(outputs.data, 1)
        
        return self.action_info_zh[predicted.cpu().item()]