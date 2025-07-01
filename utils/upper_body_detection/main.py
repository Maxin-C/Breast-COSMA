from mmpose.apis import MMPoseInferencer

class FrameDataset():
    def __init__(self, image_path, image_info):
        self.image_path = image_path

        self.frames = self._get_frames(
            cols=image_info['cols'], 
            rows=image_info['rows'], 
            total_frames=image_info['total_frames'])

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

    def get_model_input(self):
        return self.frames

class UpperBodyDetection():
    def __init__(
        self, 
        mmpose_config_path="utils/pose_estimation/mmpose_config.json",
        device="cuda:0"):
        self.device = device

        mmpose_config = SimpleNamespace(**json.load(open(mmpose_config_path,'r')))
        self.sk_inferencer = MMPoseInferencer(
            pose2d=mmpose_config.pose2d_config,
            pose2d_weights=mmpose_config.pose2d_checkpoint
            device=device
        )

    def inference(self, image_path, image_info):
        frames = FrameDataset(image_path, image_info)
        pass