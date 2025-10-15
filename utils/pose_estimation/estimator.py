import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import os
from types import SimpleNamespace
from typing import Optional, Tuple, Union, List, Dict

from mmpose.apis import MMPoseInferencer
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from transformers.utils import logging

logger = logging.get_logger("transformers")
logger.setLevel(logging.ERROR)


class MotionToPromptProjector(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim, activation='gelu'):
        super().__init__()
        self.compress = nn.Linear(feature_dim, bottleneck_dim)
        self.activation_type = activation.lower()
        if self.activation_type not in ['relu', 'gelu', 'tanh']:
            raise ValueError("激活函数必须是 'relu', 'gelu' 或 'tanh'")
        self.recover = nn.Linear(bottleneck_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, M):
        x = self.compress(M)
        if self.activation_type == 'relu':
            x = F.relu(x)
        elif self.activation_type == 'gelu':
            x = F.gelu(x)
        else:
            x = torch.tanh(x)
        P = self.recover(x)
        P = self.layer_norm(P)
        return P

class SkeletonTemporalEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_projection = nn.Linear(hidden_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.relu(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        projected = self.output_projection(lstm_out)
        final_features = self.layer_norm(projected)
        return final_features

class GuidedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, tgt, memory):
        return self.decoder_layer(tgt, memory)

class GuidedSpatialTransformer(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory)
        return output

class CLIPTextTransformerCustom(nn.Module):
    def __init__(self, text_encoder_or_config):
        super().__init__()
        if isinstance(text_encoder_or_config, nn.Module):
            self.config = text_encoder_or_config.config
            self.embeddings = text_encoder_or_config.embeddings
            self.encoder = text_encoder_or_config.encoder
            self.final_layer_norm = text_encoder_or_config.final_layer_norm
        else:
            self.config = text_encoder_or_config
            raise NotImplementedError("Initializing from config is not implemented.")
        self.eos_token_id = self.config.eos_token_id
        self._use_flash_attention_2 = getattr(self.config, "_flash_attn_2_enabled", False)

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, motion_prompt: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)
        if input_ids is None: raise ValueError("You have to specify input_ids")
        effective_batch_size, text_sequence_length = input_ids.shape
        text_embeds = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        inputs_embeds = text_embeds
        final_attention_mask = attention_mask
        prompt_len = 0
        if motion_prompt is not None:
            prompt_len = motion_prompt.shape[1]
            inputs_embeds = torch.cat([motion_prompt, text_embeds], dim=1)
            if attention_mask is not None:
                prompt_attention_mask = torch.ones(effective_batch_size, prompt_len, dtype=attention_mask.dtype, device=attention_mask.device)
                final_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            else:
                final_attention_mask = torch.ones(effective_batch_size, prompt_len + text_sequence_length, device=inputs_embeds.device, dtype=torch.long)
        current_sequence_length = inputs_embeds.shape[1]
        encoder_attention_mask_4d = None
        if self._use_flash_attention_2:
            encoder_attention_mask_4d = final_attention_mask
        elif final_attention_mask is not None:
            encoder_attention_mask_4d = _prepare_4d_attention_mask(final_attention_mask, inputs_embeds.dtype, tgt_len=current_sequence_length)
        causal_input_shape = (effective_batch_size, current_sequence_length)
        causal_attention_mask_4d = _create_4d_causal_attention_mask(causal_input_shape, inputs_embeds.dtype, device=inputs_embeds.device)
        encoder_outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=encoder_attention_mask_4d, causal_attention_mask=causal_attention_mask_4d, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        last_hidden_state = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        if self.eos_token_id is not None:
            eos_mask = (input_ids == self.eos_token_id)
            eos_indices_in_input_ids = eos_mask.int().argmax(dim=-1)
        else:
            eos_indices_in_input_ids = torch.full((effective_batch_size,), text_sequence_length - 1, device=input_ids.device, dtype=torch.long)
        eos_indices_in_concatenated_sequence = eos_indices_in_input_ids + prompt_len
        pooled_output = last_hidden_state[torch.arange(effective_batch_size, device=last_hidden_state.device), eos_indices_in_concatenated_sequence]
        if not return_dict:
            return (last_hidden_state, pooled_output) + (encoder_outputs[1:] if not isinstance(encoder_outputs, BaseModelOutput) else (encoder_outputs.hidden_states, encoder_outputs.attentions))
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class CrossAttentionWithResidual(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        output = self.norm(query + attn_output)
        return output

class MCB(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.sma = CrossAttentionWithResidual(embed_dim, num_heads)
        self.saa = CrossAttentionWithResidual(embed_dim, num_heads)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)

    def forward(self, V, T):
        V_orig, T_orig = V, T
        T_sma = self.sma(query=T, key=V, value=V)
        V_saa = self.saa(query=V, key=T, value=T)
        V_out = self.norm_v(V_orig + V_saa)
        T_out = self.norm_t(T_orig + T_sma)
        return V_out, T_out

class ClipSeq(nn.Module):
    def __init__(self, clip_model, model_config, num_actual_classes):
        super().__init__()
        for param in clip_model.parameters(): param.requires_grad = False
        self.feature_dim = clip_model.text_projection.out_features
        self.num_actual_classes = num_actual_classes
        self.image_encoder = clip_model.vision_model
        self.image_projection = clip_model.visual_projection
        self.skeleton_lstm = SkeletonTemporalEncoder(input_dim=13*3+4, embed_dim=self.feature_dim, hidden_dim=model_config.sk_hidden_dim, num_layers=model_config.sk_num_layers)
        guided_layer = GuidedTransformerLayer(d_model=self.feature_dim, nhead=model_config.st_num_heads, dim_feedforward=model_config.st_hidden_dim, dropout=model_config.st_dropout)
        self.guided_spatial_transformer = GuidedSpatialTransformer(layer=guided_layer, num_layers=model_config.st_layers)
        self.motion_transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=model_config.mt_num_heads, dim_feedforward=model_config.mt_hidden_dim, dropout=model_config.mt_dropout, batch_first=True, activation='gelu'), num_layers=model_config.mt_layers)
        self.encoder_output_projection = nn.Linear(2 * self.feature_dim, self.feature_dim)
        self.text_encoder_orig = clip_model.text_model
        self.text_encoder_cus = CLIPTextTransformerCustom(self.text_encoder_orig)
        self.text_projection = clip_model.text_projection
        self.mcb = MCB(embed_dim=self.feature_dim, num_heads=model_config.mcb_num_heads)
        self.motion_adapter = MotionToPromptProjector(feature_dim=self.feature_dim, bottleneck_dim=model_config.ma_dim, activation='gelu')
        self.conditioning_mlp = nn.Sequential(nn.Linear(self.feature_dim * 2, self.feature_dim), nn.ReLU(), nn.LayerNorm(self.feature_dim))
        self.output_classifier = nn.Linear(self.feature_dim, 2)

    def _pairwise_subtraction_all_pairs(self, input_tensor, s):
        batch_size, T, _ = input_tensor.shape
        new_differences = []
        for step in range(s - 1):
            lag = step + 1
            for i in range(lag, T):
                new_differences.append(input_tensor[:, i, :] - input_tensor[:, i - lag, :])
        if not new_differences:
            return torch.zeros(batch_size, 1, self.feature_dim, device=input_tensor.device, dtype=input_tensor.dtype) if T == 0 else input_tensor[:, :1, :]
        return torch.stack(new_differences, dim=1)

    def forward(self, images, skeletons, texts_desc_inputs, target_class_ids):
        batch_size, num_frames, _, _, _ = images.shape
        device = images.device
        image_features_pooled = self.image_encoder(images.view(-1, 3, 224, 224)).pooler_output
        image_embedding_projected = self.image_projection(image_features_pooled)
        image_embedding_normalized = image_embedding_projected / image_embedding_projected.norm(p=2, dim=-1, keepdim=True)
        fused_video_features = image_embedding_normalized.view(batch_size, num_frames, -1)
        K_s = self.skeleton_lstm(skeletons)
        fused_video_features = self.guided_spatial_transformer(tgt=fused_video_features, memory=K_s)
        image_diffs = self._pairwise_subtraction_all_pairs(fused_video_features, 3 if num_frames >= 3 else num_frames)
        motion_features = self.motion_transformer(image_diffs)
        global_motion_context = motion_features.mean(dim=1, keepdim=True)
        expanded_global_motion_context = global_motion_context.expand(-1, num_frames, -1)
        encoder_memory = torch.cat([fused_video_features, expanded_global_motion_context], dim=-1)
        encoder_memory = self.encoder_output_projection(encoder_memory)
        V_for_mcb = encoder_memory.mean(dim=1, keepdim=True)
        global_motion_prompt_for_text = self.motion_adapter(motion_features.mean(dim=1, keepdim=True))
        target_input_ids = texts_desc_inputs.input_ids[target_class_ids].to(device)
        target_attention_mask = texts_desc_inputs.attention_mask[target_class_ids].to(device)
        text_encoder_output_pooled_target = self.text_encoder_cus(input_ids=target_input_ids, attention_mask=target_attention_mask,motion_prompt=global_motion_prompt_for_text).pooler_output
        projected_text_features_target = self.text_projection(text_encoder_output_pooled_target)
        normalized_text_features_target = projected_text_features_target / projected_text_features_target.norm(p=2, dim=-1, keepdim=True)
        T_target = normalized_text_features_target.unsqueeze(1)
        _, T_prime = self.mcb(V_for_mcb, T_target)
        T_prime_expanded = T_prime.expand(-1, num_frames, -1)
        conditioned_features = torch.cat([encoder_memory, T_prime_expanded], dim=-1)
        frame_features_conditioned = self.conditioning_mlp(conditioned_features)
        logits = self.output_classifier(frame_features_conditioned.view(-1, self.feature_dim))
        outputs = logits.view(batch_size, num_frames, 2)
        return outputs

class Estimator:
    def __init__(self, config: Dict):
        self.config = SimpleNamespace(**config)
        self.device = self.config.device if torch.cuda.is_available() else 'cpu'

        with open(self.config.label_info_path, 'r') as f:
            self.label_info = json.load(f)
        self.num_actual_classes = len(self.label_info)

        mmpose_cfg = json.load(open(self.config.mmpose_config_path, 'r'))
        self.mmpose_inferencer = MMPoseInferencer(
            pose2d=mmpose_cfg['pose2d_config'],
            pose2d_weights=mmpose_cfg['pose2d_checkpoint'],
            det_model=mmpose_cfg['det_config'],
            det_weights=mmpose_cfg['det_checkpoint'],
            pose3d=mmpose_cfg['pose3d_config'],
            pose3d_weights=mmpose_cfg['pose3d_checkpoint'],
            device=self.device
        )
        
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_path)
        clip_model = CLIPModel.from_pretrained(self.config.clip_model_path)
        
        self.model = ClipSeq(
            clip_model=clip_model,
            model_config=SimpleNamespace(**json.load(open(self.config.model_config_path, 'r'))),
            num_actual_classes=self.num_actual_classes
        )
        
        if os.path.exists(self.config.trained_model_path):
            self.model.load_state_dict(torch.load(self.config.trained_model_path, map_location=self.device))
            print(f"Successfully loaded trained model from {self.config.trained_model_path}")
        else:
            raise FileNotFoundError(f"Trained model not found at {self.config.trained_model_path}")
            
        self.model.to(self.device)
        self.model.eval()

        descs = [data['desc'] for data in self.label_info]
        self.desc_inputs = self.clip_processor(text=descs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # self.desc_inputs = {k: v.to(self.device) for k, v in self.desc_inputs.items()}

    @staticmethod
    def _calculate_angle_batch(a, b, c):
        ba = a - b
        bc = c - b
        
        dot_product = np.einsum('ij,ij->i', ba, bc)
        norm_ba = np.linalg.norm(ba, axis=1)
        norm_bc = np.linalg.norm(bc, axis=1)
        
        cosine_angle = dot_product / (norm_ba * norm_bc + 1e-7)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        return np.degrees(np.arccos(cosine_angle))

    @staticmethod
    def _calculate_upper_body_angles_batch(keypoints):
        if keypoints.shape[1:] != (17, 3):
            raise ValueError(f"输入关键点形状应为 (n_frames, 17, 3), 但得到 {keypoints.shape}")
        
        left_shoulder, right_shoulder = keypoints[:, 5], keypoints[:, 6]
        left_elbow, right_elbow = keypoints[:, 7], keypoints[:, 8]
        left_wrist, right_wrist = keypoints[:, 9], keypoints[:, 10]
        left_hip, right_hip = keypoints[:, 11], keypoints[:, 12]
        
        left_elbow_angle = Estimator._calculate_angle_batch(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = Estimator._calculate_angle_batch(right_shoulder, right_elbow, right_wrist)
        left_shoulder_angle = Estimator._calculate_angle_batch(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = Estimator._calculate_angle_batch(right_elbow, right_shoulder, right_hip)
        
        return np.stack([
            left_elbow_angle, right_elbow_angle,
            left_shoulder_angle, right_shoulder_angle
        ], axis=1)

    def _split_sprite_sheet(self, sprite_image: Image.Image) -> List[Image.Image]:
        # if sprite_image.width % 3 != 0 or sprite_image.height % 2 != 0:
        #     raise ValueError("Sprite sheet dimensions must be divisible by 3 (width) and 2 (height).")
        
        img_width = sprite_image.width // 3
        img_height = sprite_image.height // 2
        frames = []
        for i in range(2): # rows
            for j in range(3): # cols
                left = j * img_width
                top = i * img_height
                right = left + img_width
                bottom = top + img_height
                frame = sprite_image.crop((left, top, right, bottom))
                frames.append(frame)
        return frames

    def _extract_skeletons(self, frames: List[Image.Image]) -> torch.Tensor:
        frame_arrays = [np.array(frame.convert('RGB')) for frame in frames]
        
        result_generator = self.mmpose_inferencer(frame_arrays, show=False, save_results=False, batch_size=len(frames))
        results = [res for res in result_generator]
        
        keypoints_list = []
        for frame_result in results:
            predictions = frame_result.get('predictions', [])
            if not predictions or not predictions[0]:
                keypoints_list.append(np.zeros((17, 3)))
            else:
                person_data = predictions[0][0]
                keypoints = np.array(person_data['keypoints'])
                keypoints_list.append(keypoints)

        keypoints_array = np.stack(keypoints_list) # Shape: [num_frames, 17, 3]

        angles = self._calculate_upper_body_angles_batch(keypoints_array) # Shape: [num_frames, 4]

        upper_body_kpts = keypoints_array[:, :13, :].reshape(keypoints_array.shape[0], -1) # Shape: [num_frames, 39]

        skeleton_matrix = np.concatenate((upper_body_kpts, angles), axis=1) # Shape: [num_frames, 43]

        return torch.from_numpy(skeleton_matrix).unsqueeze(0).to(torch.float32)

    def predict(self, sprite_image: Image.Image, target_class_id: int) -> List[int]:
        
        with torch.no_grad():
            frames = self._split_sprite_sheet(sprite_image)
            
            skeletons_tensor = self._extract_skeletons(frames).to(self.device)
            
            images_processed = self.clip_processor(images=frames, return_tensors="pt", padding=True)
            images_tensor = images_processed['pixel_values'].to(self.device)
            images_tensor = images_tensor.unsqueeze(0) if images_tensor.dim() == 4 else images_tensor
            
            target_class_ids_tensor = torch.tensor([target_class_id], device=self.device)

            outputs = self.model(
                images=images_tensor, 
                skeletons=skeletons_tensor, 
                texts_desc_inputs=self.desc_inputs, 
                target_class_ids=target_class_ids_tensor
            )
            
            _, predictions = torch.max(outputs.data, 2)
            result_array = predictions.squeeze(0).cpu().tolist()
        
        return result_array

if __name__ == '__main__':
    config = {
        "device": "cuda:0",
        "clip_model_path": "/root/huggingface/openai/clip-vit-large-patch14",
        "trained_model_path": "utils/pose_estimation/model_dict/model_20250716_032346_acc76.46.pth",
        "label_info_path": "utils/pose_estimation/motion_desc.json",
        "model_config_path": "utils/pose_estimation/model_config.json",
        "mmpose_config_path": "utils/pose_estimation/mmpose_config.json"
    }

    classifier_service = Estimator(config)
    sample_sprite_path = "/var/codes/Breast-COMA-Rehab/uploads/slices/1_1755648889472_sprite_sheet.png"
    
    target_class_id = 1
    sample_sprite_image = Image.open(sample_sprite_path)
    results = classifier_service.predict(sample_sprite_image, target_class_id)

    for i, res in enumerate(results):
        status = "属于" if res == 1 else "不属于"
        print(f"Frame {i+1}: {status} '{target_class_id}'")
    print(f"\nRaw output array: {results}")
