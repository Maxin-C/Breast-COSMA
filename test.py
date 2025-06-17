from utils.pose_estimation.main import PoseEstimation
from tqdm import tqdm

pose_estimation = PoseEstimation(device="cuda:2")
for i in tqdm(range(10)):
    print(pose_estimation.inference(
        image_path="sprite_sheet_pil.jpg",
        image_info={
            'cols': 4,
            'rows': 2,
            'total_frames': 8
        }
    ))