from typing import List, Optional, Callable

import torch
import torch.nn.functional as F

from config import RunConfig
from constants import OUT_INDEX, STRUCT_INDEX, STYLE_INDEX
from models.stable_diffusion_final import CrossImageAttentionStableDiffusionPipeline
from utils import attention_utils
from utils.adain import masked_adain, adain
from utils.model_utils_final import get_stable_diffusion_model
from utils.segmentation import Segmentor

from einops import rearrange, repeat
import math
import numpy as np
import cv2
import os
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

class AppearanceTransferModel:

    def __init__(self, config: RunConfig, pipe: Optional[CrossImageAttentionStableDiffusionPipeline] = None):
        self.config = config
        self.pipe = get_stable_diffusion_model() if pipe is None else pipe
        self.register_attention_control()
        self.segmentor = Segmentor(prompt=config.prompt, object_nouns=[config.object_noun])
        self.latents_app, self.latents_struct = None, None
        self.zs_app, self.zs_struct = None, None
        self.image_app_mask_32, self.image_app_mask_64 = None, None
        self.image_struct_mask_32, self.image_struct_mask_64 = None, None
        self.enable_edit = False
        self.sam_app_mask, self.sam_struct_mask = None, None
        self.step = 0

    def set_latents(self, latents_app: torch.Tensor, latents_struct: torch.Tensor):
        self.latents_app = latents_app
        self.latents_struct = latents_struct

    def set_noise(self, zs_app: torch.Tensor, zs_struct: torch.Tensor):
        self.zs_app = zs_app
        self.zs_struct = zs_struct

    def set_masks(self, masks: List[torch.Tensor]):
        self.image_app_mask_32, self.image_struct_mask_32, self.image_app_mask_64, self.image_struct_mask_64 = masks
        
    def set_masks_32(self, masks: List[torch.Tensor]):
        self.image_app_mask_32, self.image_struct_mask_32 = masks
        self.image_app_mask_32 = self.image_app_mask_32.float().to(self.latents_app.device)
        self.image_struct_mask_32 = self.image_struct_mask_32.float().to(self.latents_app.device)

        self.image_app_mask_64 = F.interpolate(self.image_app_mask_32.unsqueeze(dim=0), size=(64, 64), mode='nearest').squeeze()
        self.image_struct_mask_64 = F.interpolate(self.image_struct_mask_32.unsqueeze(dim=0), size=(64, 64), mode='nearest').squeeze()
    
    def mask_down(self, mask: List[torch.Tensor]):
        mask = F.interpolate(torch.tensor(mask).float().view(1,1,512,512), size=(64, 64), mode='bilinear').view(64, 64).to(self.latents_app.device)
        return mask

    def get_adain_callback(self):

        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> Callable:
            self.step = st
            if self.config.use_masked_adain and self.step == self.config.adain_range.start and self.config.mask_use==False:
                masks = self.segmentor.get_object_masks()
                self.set_masks(masks)
            
            if self.config.feat_range.start <= self.step < self.config.feat_range.end and self.config.do_cross_mask==True:
                masks = self.segmentor.cross_attn_map(thres=self.config.cross_thres)
                self.set_masks_32(masks)

            # Apply AdaIN operation using the computed masks
            if self.config.adain_range.start-1 <= self.step < self.config.adain_range.end+1:                
                if self.config.use_masked_adain:
                    if self.config.mask_use==False:
                        latents[0] = masked_adain(latents[0], latents[1], self.image_struct_mask_64, self.image_app_mask_64)
                    else:
                        self.image_struct_mask_64 = self.mask_down(self.sam_struct_mask)
                        self.image_app_mask_64 = self.mask_down(self.sam_app_mask)
                        latents[0] = masked_adain(latents[0], latents[1], self.image_struct_mask_64, self.image_app_mask_64)
                else:
                    latents[0] = adain(latents[0], latents[1]) 

        return callback

    def register_attention_control(self):

        model_self = self

        class AttentionProcessor:

            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet
                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError("AttnProcessor2_0 requires torch 2.0, to use it, please upgrade torch to 2.0.")
            
            def get_new_filename(self, filename, folder_path):
                base_name, ext = os.path.splitext(filename)
                count = 1
                new_filename = filename
                while os.path.exists(os.path.join(folder_path, new_filename)):
                    new_filename = f"{base_name}_{count}{ext}"
                    count += 1
                final_pth = os.path.join(folder_path, new_filename)
                return final_pth
            
            def change_feature(self, x, src_lst, match_lst):
                feature_h, feature_w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))

                x = rearrange(x, 'b (h w) c -> b h w c', b=3, h=feature_h) # b=[out, style, struct]
                for i in range(0, len(src_lst)):
                    x[OUT_INDEX][src_lst[i][1], src_lst[i][0], :] = x[STYLE_INDEX][match_lst[i][1], match_lst[i][0],:] 
                    
                x = rearrange(x, 'b h w c -> b (h w) c')
                
                return x            
            
            def matching_feature(self, x, only_match_out=False, mask_lst=None):
                feature_h, feature_w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
                if mask_lst:
                    style_mask = F.interpolate(torch.tensor(mask_lst[0][0]).float().view(1,1,512,512), size=(feature_h, feature_w), mode='bilinear').view(feature_h, feature_w).unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x.device)
                    struct_mask = F.interpolate(torch.tensor(mask_lst[1][0]).float().view(1,1,512,512), size=(feature_h, feature_w), mode='bilinear').view(feature_h, feature_w).unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x.device)
                    
                    style_mask = rearrange(style_mask, 'h w b -> b h w') 
                    struct_mask = rearrange(struct_mask, 'h w b -> b h w') 
                     
                src_lst = []
                match_lst = []
                x = rearrange(x, 'b (h w) c -> b c h w', b=x.shape[0], h=feature_h) # b=[out, style, struct]
                
                if x.shape[0]==6:
                    OUT_INDEX=3
                    STYLE_INDEX=4
                
                out_x = x[OUT_INDEX] #[c,h,w]
                style_x = x[STYLE_INDEX] 
                if mask_lst:
                    out_x = out_x * struct_mask
                    style_x = style_x * style_mask
                
                for x_coor, y_coor in np.ndindex(feature_h, feature_w):
                    out_vec = out_x[:, y_coor, x_coor].view(1, -1)  # 1, C
                    if torch.all(out_vec == 0):
                        pass 
                    else:
                        style_vec = rearrange(style_x, 'c h w -> c (h w)', h=feature_h).unsqueeze(dim=0)
                        
                        out_vec = F.normalize(out_vec) # 1, C
                        style_vec = F.normalize(style_vec) # 1, C, HW
                        cos_map = torch.matmul(out_vec, style_vec).view(1, feature_h, feature_h).cpu().numpy() # N, H, W
                        
                        max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)

                        if only_match_out==False:
                            x[OUT_INDEX][:,y_coor, x_coor] = x[STYLE_INDEX][:,max_yx[0].item(), max_yx[1].item()]
                        src_lst.append((x_coor, y_coor))
                        match_lst.append((max_yx[1].item(), max_yx[0].item()))

                x = rearrange(x, 'b c h w -> b (h w) c')
                return x, src_lst, match_lst

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         v_swap: bool = False,
                         feature_swap: bool = False,
                         check_rgb=True,
                         mask_lst=None
                         ):
                
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                is_cross = encoder_hidden_states is not None

                if v_swap:
                    hidden_states, src_lst, match_lst = self.matching_feature(hidden_states, only_match_out=True, mask_lst=mask_lst)
                # Feature injection
                if "up" in self.place_in_unet and feature_swap and not is_cross:                      
                    if attention_utils.should_mix_features_mix(model_self, hidden_states):
                        if model_self.step % 5 == 0 and model_self.step < 40:
                            hidden_states[OUT_INDEX] = hidden_states[STRUCT_INDEX]
                        else:
                            hidden_states, src_lst, match_lst = self.matching_feature(hidden_states, only_match_out=False, mask_lst=mask_lst)


                query = attn.to_q(hidden_states)
            
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False

                # V injection
                if v_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                    if attention_utils.should_mix_features_mix(model_self, hidden_states):
                        should_mix = True
                        value = self.change_feature(value, src_lst, match_lst)
                        #key = self.change_feature(key, src_lst, match_lst) # you can use this line for KV injection

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # Compute the cross attention and apply our contrasting operation
                hidden_states, attn_weight = attention_utils.compute_scaled_dot_product_attention(
                    query, key, value,
                    edit_map=v_swap and model_self.enable_edit and should_mix,
                    is_cross=is_cross,
                    contrast_strength=model_self.config.contrast_strength,
                )                                            

                if model_self.config.use_masked_adain:
                    model_self.segmentor.update_attention(attn_weight, is_cross)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
            if net_.__class__.__name__ == 'Attention':
                net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}"))
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.pipe.unet.named_children()

        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
