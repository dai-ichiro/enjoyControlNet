import os
os.makedirs('results', exist_ok=True)

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, EulerAncestralDiscreteScheduler

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed',
    type=int,
    default=200,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=1,
    help='how many samples to produce for each given prompt',
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
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='path to original image'
)
args = parser.parse_args()

seed = args.seed
steps = args.steps

bool_vae = False if args.vae is None else True
vae_folder = args.vae

if bool_vae:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_folder).to("cuda")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "control_counterfeit_canny",
        vae=vae,
        safety_checker=None).to("cuda")
else:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "control_counterfeit_canny",
        safety_checker=None).to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

original_image =Image.open(args.image)
control = cv2.Canny(np.array(original_image), threshold1=100, threshold2=200)

Image.fromarray(control).save(os.path.join('results', 'canny.png'))

for i in range(args.n_samples):
    seed_i = seed + i * 1000
    generator = torch.manual_seed(seed_i)
    image = pipe(
        prompt="best quality, extremely detailed", 
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        controlnet_hint=control,
        num_inference_steps=steps, 
        generator=generator).images[0]

    if bool_vae:
        image.save(os.path.join('results', f'seed{seed_i}_vae.png'))
    else:
        image.save(os.path.join('results', f'seed{seed_i}.png'))

