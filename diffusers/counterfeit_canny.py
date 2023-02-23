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
args = parser.parse_args()

seed = args.seed
steps = args.steps

threshold1 = args.threshold1
threshold2 = args.threshold2

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

if args.from_canny:
    control = np.array(Image.open(args.image))
else:
    control = cv2.Canny(np.array(Image.open(args.image)), threshold1=threshold1, threshold2=threshold2)

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

