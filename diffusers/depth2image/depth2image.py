import os 
os.makedirs('results', exist_ok=True)

import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, AutoencoderKL
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
    '--from_depth',
    action="store_true",
    help='if true, use depth image'
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

vae_folder =args.vae
base_model_id = args.model

if args.from_scribble:
    controlhint = load_image(args.image)
else:
    #todo 
    controlhint = load_image(args.image)

if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae').to('cuda')

controlnet = ControlNetModel.from_pretrained("basemodel/sd-controlnet-scribble")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
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
            image=controlhint,
            num_inference_steps=steps, 
            generator=generator,
            guidance_scale = scale,
            ).images[0]

        image.save(os.path.join('results', f'scale{scale}_seed{seed_i}.png'))
    
