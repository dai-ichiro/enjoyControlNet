import os 
os.makedirs('results', exist_ok=True)

import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
import controlnet_hinter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='model',
)
parser.add_argument(
    '--seed',
    type=int,
    default=20000,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=1,
    help='how many samples to produce for each given prompt',
)
parser.add_argument(
    '--scale',
    nargs='*',
    default=[9.0],    
    type=float,
    help='guidance_scale',
)
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='path to original image'
)
parser.add_argument(
    '--threshold1',
    type=int,
    default=100,
    help='low_threshold'
)
parser.add_argument(
    '--threshold2',
    type=int,
    default=200,
    help='high_threshold'
)
parser.add_argument(
    '--from_canny',
    action="store_true",
    help='if true, use canny image'
)
parser.add_argument(
    '--steps',
    default=30,
    type=int,
    help='num_inference_steps',
)
parser.add_argument(
    '--vae',
    type=str,
    help='vae'
)
args = parser.parse_args()

seed = args.seed
steps = args.steps
scale_list = args.scale

threshold1 = args.threshold1
threshold2 = args.threshold2


vae_folder =args.vae
base_model_id = args.model

if args.from_canny:
    control = load_image(args.image)
else:
    control = controlnet_hinter.hint_canny(np.array(load_image(args.image)), low_threshold=threshold1, high_threshold=threshold2)
    control.save(os.path.join('results', 'canny.png'))

if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae').to('cuda')

unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder='unet').to('cuda')

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    'basemodel/control_sd15_canny',
    unet=unet,
    vae=vae,
    safety_checker=None).to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

for i in range(args.n_samples):
    seed_i = seed + i * 1000
    for scale in scale_list:
        generator = torch.manual_seed(seed_i)
        image = pipe(
            prompt="best quality, extremely detailed", 
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            image=control,
            width = 512,
            height = 512,
            num_inference_steps=steps, 
            generator=generator,
            guidance_scale = scale,
            ).images[0]

        image.save(os.path.join('results', f'scale{scale}_seed{seed_i}.png'))
    
