import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
import gc
import torch.nn as nn
import math
import cv2
import os
import re
import matplotlib.pyplot as plt
import json

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model_final import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images
from sam_hq.segment_anything import sam_model_registry, SamPredictor

@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
 
    
def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()
        
def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def get_file_name(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    number = numbers[0]
    return number
    
def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    model = AppearanceTransferModel(cfg)
    latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cfg)
    model.set_latents(latents_app, latents_struct)
    model.set_noise(noise_app, noise_struct)
    print("Running appearance transfer...")
    images = run_appearance_transfer(model=model, cfg=cfg)
    print("Done.")
    return images


def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    mask_lst=None
    if cfg.mask_use: #cfg.sam_path
        if cfg.bbox_path:
            bbox=load_json(cfg.bbox_path)
        model_type = "vit_l"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=cfg.sam_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        app_image = cv2.imread(str(cfg.app_image_path))
        app_image = cv2.resize(app_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        app_image = cv2.cvtColor(app_image, cv2.COLOR_BGR2RGB)
        
        struct_image = cv2.imread(str(cfg.struct_image_path))
        struct_image = cv2.resize(struct_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        struct_image = cv2.cvtColor(struct_image, cv2.COLOR_BGR2RGB)
        imgs = [app_image, struct_image]
        
        mask_lst=[]
        for i in range(0, len(imgs)):
            predictor.set_image(imgs[i])
            hq_token_only = True

            input_box = np.array([[0, 0, 512, 512]])
            if cfg.bbox_path:
                if i==0:
                    app_name = get_file_name(cfg.app_image_path)
                    input_box = bbox.get(app_name, [])
                else:
                    struct_name = get_file_name(cfg.struct_image_path)
                    input_box = bbox.get(struct_name, [])
                input_box = np.array([input_box])
            else:
                input_box = np.array([[0, 0, 512, 512]]) # Cover Full Image

            input_point, input_label = None, None
            result_path = str(cfg.rgb_check_path)
            masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box = input_box,
                    multimask_output=False,
                    hq_token_only=hq_token_only, 
                )
            show_res(masks,scores,input_point, input_label, input_box, result_path + '/mask' + str(i), imgs[i])
            mask_lst.append(masks) #0:style, 1:struct
            
        model.sam_app_mask = mask_lst[0]
        model.sam_struct_mask = mask_lst[1]

    images = model.pipe(
        prompt=[cfg.prompt] * 3,
        latents=init_latents,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
        mask_lst = mask_lst,
        do_v_swap = cfg.do_v_swap
    ).images
    
    # Save images
    images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}.png")
    images[1].save(cfg.output_path / f"out_style---seed_{cfg.seed}.png")
    images[2].save(cfg.output_path / f"out_struct---seed_{cfg.seed}.png")
    joined_images = np.concatenate(images[::-1], axis=1)
    Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    
    return images


if __name__ == '__main__':
    main()