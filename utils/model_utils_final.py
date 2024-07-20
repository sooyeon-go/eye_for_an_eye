import torch
from diffusers import DDIMScheduler

from models.stable_diffusion_final import CrossImageAttentionStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel

def get_stable_diffusion_model() -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    try:
        with open('./TOKEN', 'r') as f:
            token = f.read().replace('\n', '') # remove the last \n!
            print(f'[INFO] loaded hugging face access token from ./TOKEN!')
    except FileNotFoundError as e:
        token = True
        print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                      safety_checker=None, use_auth_token=token).to(device)
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", use_auth_token=token).to(device)
    pipe.scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    print("Done.")
    return pipe