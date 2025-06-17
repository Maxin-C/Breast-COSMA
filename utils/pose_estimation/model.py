import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from transformers.utils import logging

import math
import numpy as np
from typing import Optional, Tuple, Union

class MotionToPromptProjector(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim, activation='gelu'):
        super().__init__()
        self.compress = nn.Linear(feature_dim, bottleneck_dim)
        self.activation_type = activation.lower()
        if self.activation_type not in ['relu', 'gelu', 'tanh']:
            raise ValueError("激活函数必须是 'relu', 'gelu' 或 'tanh'")
        self.recover = nn.Linear(bottleneck_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim) # 添加 LayerNorm

    def forward(self, M):
        x = self.compress(M)
        if self.activation_type == 'relu':
            x = F.relu(x)
        elif self.activation_type == 'gelu':
            x = F.gelu(x)
        else:
            x = torch.tanh(x)
        P = self.recover(x)
        P = self.layer_norm(P) # 应用 LayerNorm
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

class FusionModule(nn.Module):
    def __init__(self, rgb_dim, skel_dim, fused_dim):
        super().__init__()
        self.fusion_layer = nn.Linear(rgb_dim + skel_dim, fused_dim)
        self.layer_norm = nn.LayerNorm(fused_dim)
        self.relu = nn.ReLU()

    def forward(self, f_rgb, f_skel):
        f_cat = torch.cat([f_rgb, f_skel], dim=-1)
        f_fused = self.relu(self.fusion_layer(f_cat))
        f_fused = self.layer_norm(f_fused)
        return f_fused

class CLIPTextTransformerCustom(nn.Module):
    def __init__(self, text_encoder_or_config): # 可以传入完整的text_encoder或其config
        super().__init__()
        if isinstance(text_encoder_or_config, nn.Module): # 如果是text_encoder实例
            self.config = text_encoder_or_config.config
            self.embeddings = text_encoder_or_config.embeddings
            self.encoder = text_encoder_or_config.encoder
            self.final_layer_norm = text_encoder_or_config.final_layer_norm
        else:
            self.config = text_encoder_or_config
            raise NotImplementedError("Initializing CLIPTextTransformerCustom from config is not fully implemented here. Pass the text_encoder module.")

        self.eos_token_id = self.config.eos_token_id
        self._use_flash_attention_2 = getattr(self.config, "_flash_attn_2_enabled", False) or \
                                     getattr(self.config, "use_flash_attention_2", False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        motion_prompt: Optional[torch.Tensor] = None, # [effective_batch_size, prompt_seq_len, hidden_size], effective_batch_size = batch_size * class_num
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D (batch_size, sequence_length), got {input_ids.shape}")
        
        effective_batch_size, text_sequence_length = input_ids.shape
        
        text_embeds = self.embeddings(input_ids=input_ids, position_ids=position_ids) # [EBS, text_seq_len, hidden_size]

        inputs_embeds = text_embeds
        final_attention_mask = attention_mask
        prompt_len = 0

        if motion_prompt is not None:
            if motion_prompt.shape[0] != effective_batch_size:
                raise ValueError(
                    f"Batch size mismatch: motion_prompt has {motion_prompt.shape[0]}, "
                    f"input_ids has {effective_batch_size}"
                )
            if motion_prompt.ndim != 3 or motion_prompt.shape[2] != text_embeds.shape[2]:
                raise ValueError(
                    f"motion_prompt must be 3D (batch, prompt_seq_len, hidden_size) and match hidden_size, "
                    f"got {motion_prompt.shape} and text_embeds hidden_size {text_embeds.shape[2]}"
                )
            
            prompt_len = motion_prompt.shape[1]
            inputs_embeds = torch.cat([motion_prompt, text_embeds], dim=1) # [EBS, prompt_len + text_seq_len, hidden_size]
            
            if attention_mask is not None: # [EBS, text_seq_len]
                prompt_attention_mask = torch.ones(
                    effective_batch_size,
                    prompt_len,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                final_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            else:
                final_attention_mask = torch.ones(
                    effective_batch_size,
                    prompt_len + text_sequence_length,
                    device=inputs_embeds.device,
                    dtype=torch.long
                )
        
        current_sequence_length = inputs_embeds.shape[1]
        
        encoder_attention_mask_4d = None
        if self._use_flash_attention_2:
             encoder_attention_mask_4d = final_attention_mask 
        elif final_attention_mask is not None :
            encoder_attention_mask_4d = _prepare_4d_attention_mask(final_attention_mask, inputs_embeds.dtype, tgt_len=current_sequence_length)

        causal_input_shape = (effective_batch_size, current_sequence_length)
        causal_attention_mask_4d = _create_4d_causal_attention_mask(
            causal_input_shape, inputs_embeds.dtype, device=inputs_embeds.device
        )
        
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask_4d, # 2D for FlashAttn2, 4D otherwise
            causal_attention_mask=causal_attention_mask_4d, # Always 4D, this is combined with attention_mask by HF's CLIPEncoder
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id is not None:
            eos_mask = (input_ids == self.eos_token_id)
            eos_indices_in_input_ids = eos_mask.int().argmax(dim=-1)

        else:
            eos_indices_in_input_ids = torch.full(
                (effective_batch_size,), text_sequence_length - 1, device=input_ids.device, dtype=torch.long
            )

        eos_indices_in_concatenated_sequence = eos_indices_in_input_ids + prompt_len

        pooled_output = last_hidden_state[
            torch.arange(effective_batch_size, device=last_hidden_state.device),
            eos_indices_in_concatenated_sequence,
        ]
        
        if not return_dict:
            outputs = (last_hidden_state, pooled_output) + \
                      (encoder_outputs[1:] if not isinstance(encoder_outputs, BaseModelOutput) else (encoder_outputs.hidden_states, encoder_outputs.attentions))
            return outputs

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class CrossAttentionWithResidual(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # MultiheadAttention期望 (seq_len, batch, embed_dim) 或 (batch, seq_len, embed_dim) if batch_first=True
        # PyTorch 默认 batch_first=False for MultiheadAttention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # 设置 batch_first=True
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value):
        # 输入 query, key, value 期望是 (batch, seq_len, embed_dim)
        assert query.dim() == 3 and key.dim() == 3 and value.dim() == 3, "Inputs must be 3D"

        attn_output, _ = self.attention(query, key, value) # attn_output: (batch, query_seq_len, embed_dim)
        
        # 残差连接和归一化
        output = self.norm(query + attn_output)
        return output

class MCB(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.sma = CrossAttentionWithResidual(embed_dim, num_heads)
        self.saa = CrossAttentionWithResidual(embed_dim, num_heads)
        self.norm_v = nn.LayerNorm(embed_dim) # 为V和T分别使用LayerNorm可能更好
        self.norm_t = nn.LayerNorm(embed_dim)
        
    def forward(self, V, T):
        assert V.dim() == 3, f"V should be 3D, got {V.shape}" # V: [batch_size, 1, feature_dim]
        assert T.dim() == 3, f"T should be 3D, got {T.shape}" # T: [batch_size, num_classes, feature_dim]

        V_orig = V
        T_orig = T
        
        # SMA: T as query, V as key/value
        # query: T [bs, N_classes, D], key/value: V [bs, 1, D]
        T_sma = self.sma(query=T, key=V, value=V) # Output: [bs, N_classes, D]
        
        # SAA: V as query, T as key/value
        # query: V [bs, 1, D], key/value: T [bs, N_classes, D]
        V_saa = self.saa(query=V, key=T, value=T) # Output: [bs, 1, D]
        
        V_out = self.norm_v(V_orig + V_saa)
        T_out = self.norm_t(T_orig + T_sma)
        
        return V_out, T_out

class PoseEstimationModel(nn.Module):
    def __init__(self, clip_model: CLIPModel, model_config):
        super().__init__()

        for param in clip_model.parameters():
            param.requires_grad = False

        self.image_encoder = clip_model.vision_model
        self.text_encoder_orig = clip_model.text_model 
        self.image_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection 
        
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        self.feature_dim = self.image_projection.out_features 

        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=model_config.st_num_heads,
                dim_feedforward=model_config.st_hidden_dim,
                dropout=0.1, 
                batch_first=True,
                activation='gelu' 
            ),
            num_layers=model_config.st_layers
        )
        self.motion_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=model_config.mt_num_heads,
                dim_feedforward=model_config.mt_hidden_dim,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=model_config.mt_layers
        )
        self.skeleon_lstm = SkeletonTemporalEncoder(
            input_dim=13*3+4, 
            embed_dim=model_config.sk_embed_dim, 
            hidden_dim=model_config.sk_hidden_dim, 
            num_layers=model_config.sk_num_layers
        )
        self.fusion_module = FusionModule(
            rgb_dim=self.feature_dim, 
            skel_dim=model_config.sk_embed_dim, 
            fused_dim=self.feature_dim
        )

        self.motion_adapter = MotionToPromptProjector(
            feature_dim=self.feature_dim,
            bottleneck_dim=model_config.ma_dim,
            activation='gelu'
        )
        self.text_encoder_cus = CLIPTextTransformerCustom(self.text_encoder_orig)

        self.mcb = MCB(embed_dim=self.feature_dim, num_heads=model_config.mcb_num_heads)

    def _pairwise_subtraction_all_pairs(self, input_tensor, s):
        batch_size, T, _ = input_tensor.shape # seq_len is feature_dim
        
        new_differences = []
        for step in range(s - 1):  # step = 0, 1 for s=3 (corresponds to lag = 1, 2)
            lag = step + 1
            for i in range(lag, T): # Corrected loop for clarity: i is the time index of the minuend
                new_differences.append(input_tensor[:, i, :] - input_tensor[:, i - lag, :])
        
        if not new_differences: 
            print(f"Warning: No differences generated in _pairwise_subtraction_all_pairs for T={T}, s={s}. Returning zeros.")
            original_diffs = []
            for step_orig in range(s - 1):
                for i_orig in range(T):
                    if i_orig - (step_orig + 1) >= 0:
                        original_diffs.append(input_tensor[:, i_orig, :] - input_tensor[:, i_orig - (step_orig + 1), :])
            if not original_diffs: # Should not happen if T is reasonable (e.g. T >= s)
                 return torch.zeros(batch_size, 1, self.feature_dim, device=input_tensor.device, dtype=input_tensor.dtype) # Placeholder if empty
            return torch.stack(original_diffs, dim=1)

        return torch.stack(new_differences, dim=1)


    def forward(self, images, skeleons, texts_desc_inputs): # texts_desc_inputs is clip_processor output
        # Image Encoder
        batch_size = images.shape[0]
        image_num = images.shape[1] # e.g., 8 frames

        image_features_pooled = self.image_encoder(images.view(-1, 3, 224, 224)).pooler_output
        image_embedding_projected = self.image_projection(image_features_pooled) 
        image_embedding_normalized = image_embedding_projected / image_embedding_projected.norm(p=2, dim=-1, keepdim=True)
        image_embed_sequence = image_embedding_normalized.view(batch_size, image_num, -1) # [batch_size, 8, feature_dim]

        # Spatial Transformer
        T_s = self.spatial_transformer(image_embed_sequence) # [batch_size, 8, feature_dim]

        # Skeleon LSTM
        K_s = self.skeleon_lstm(skeleons)

        # Fusion
        T_s = self.fusion_module(T_s, K_s)

        # Motion Transformer
        # Assuming image_num (T) is >= 3 for s=3.
        if image_num < 3 and image_num >0 : # s=3 implies min T=3 for pairwise subtraction of lag 2
             # Handle case where not enough frames for desired subtractions.
             # Fallback: use simpler motion features or replicate T_s if T_m cannot be computed.
             print(f"Warning: image_num={image_num} is less than s=3. Motion features might be simplified.")
             # Example fallback: use original embeddings for T_m or a zero tensor of expected shape
             if image_num > 1:
                image_diffs_simple = self._pairwise_subtraction_all_pairs(image_embed_sequence, image_num) # Max possible s = image_num
             else: # if image_num == 1
                image_diffs_simple = torch.zeros(batch_size, 1, self.feature_dim, device=images.device, dtype=images.dtype) # Placeholder for T_m input
             
             if image_diffs_simple.shape[1] == 0: # If still no diffs (e.g. T=1)
                 T_m_features = torch.zeros(batch_size, 1, self.feature_dim, device=images.device, dtype=images.dtype) # Default T_m
             else:
                 T_m_features = self.motion_transformer(image_diffs_simple)
        elif image_num == 0: # No images
            T_s = torch.zeros(batch_size, 1, self.feature_dim, device=images.device, dtype=images.dtype) # Placeholder for T_s
            T_m_features = torch.zeros(batch_size, 1, self.feature_dim, device=images.device, dtype=images.dtype) # Placeholder for T_m
        else: # Sufficient images
            image_diffs = self._pairwise_subtraction_all_pairs(image_embed_sequence, 3) # [batch_size, 13, feature_dim]
            T_m_features = self.motion_transformer(image_diffs) # [batch_size, 13, feature_dim]


        # Mean Pooling for V
        V = (T_s.mean(dim=1, keepdim=True) + T_m_features.mean(dim=1, keepdim=True))


        # Motion Adapter -> motion_prompt
        motion_prompt = self.motion_adapter(T_m_features) # [batch_size, motion_seq_len (e.g. 13), feature_dim]

        # Text Processing
        # texts_desc_inputs.input_ids: [num_classes, text_seq_len]
        # texts_desc_inputs.attention_mask: [num_classes, text_seq_len]
        num_classes = texts_desc_inputs.input_ids.size(0)
        text_seq_len = texts_desc_inputs.input_ids.size(1)
        motion_seq_len = motion_prompt.size(1)

        # Expand input_ids and attention_mask for each item in batch
        # target: [batch_size * num_classes, text_seq_len]
        flat_input_ids = texts_desc_inputs.input_ids.unsqueeze(0).expand(batch_size, num_classes, text_seq_len).reshape(batch_size * num_classes, text_seq_len)
        flat_attention_mask = texts_desc_inputs.attention_mask.unsqueeze(0).expand(batch_size, num_classes, text_seq_len).reshape(batch_size * num_classes, text_seq_len)

        # Expand motion_prompt for each class description
        # target: [batch_size * num_classes, motion_seq_len, feature_dim]
        flat_motion_prompt = motion_prompt.unsqueeze(1).expand(batch_size, num_classes, motion_seq_len, self.feature_dim).reshape(batch_size * num_classes, motion_seq_len, self.feature_dim)

        # Get text embeddings using custom text encoder
        text_encoder_output_pooled = self.text_encoder_cus(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            motion_prompt=flat_motion_prompt
        ).pooler_output # Output: [batch_size * num_classes, feature_dim]

        # !! Crucial: Apply the original CLIP text_projection layer !!
        projected_text_features = self.text_projection(text_encoder_output_pooled)
        normalized_text_features = projected_text_features / projected_text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Reshape T to [batch_size, num_classes, feature_dim]
        T = normalized_text_features.view(batch_size, num_classes, -1)
        
        # MCB Module
        V_prime, T_prime = self.mcb(V, T) # V_prime: [bs,1,D], T_prime: [bs,N_classes,D]

        # Probability Calculation
        # scaled_scores = F.cosine_similarity(V_prime, T_prime, dim=-1) / self.tau # Original fixed tau
        
        # Use learnable logit_scale
        logit_scale_exp = self.logit_scale.exp()
        scaled_scores = F.cosine_similarity(V_prime, T_prime, dim=2) * logit_scale_exp
        
        return scaled_scores # [batch_size, num_classes]