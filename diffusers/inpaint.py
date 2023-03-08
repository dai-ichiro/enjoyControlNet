import os
import torch
from stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline

from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--controlnet',
    type=str,
    required=True,
    help='controlnet'
)
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='original image'
)
parser.add_argument(
    '--mask',
    type=str,
    required=True,
    help='mask image'
)
parser.add_argument(
    '--hint',
    type=str,
    required=True,
    help='controlnet hint image'
)
parser.add_argument(
    '--W',
    type=int,
    default=512,
    help='width'
)
parser.add_argument(
    '--H',
    type=int,
    default=512,
    help='height'
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
    '--prompt',
    type=str,
    help='prompt'
)
opt = parser.parse_args()

width = opt.W
height = opt.H

controlnet_model = opt.controlnet

controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "model/stable-diffusion-inpainting", 
    controlnet=controlnet, 
    safety_checker=None, 
    torch_dtype=torch.float16)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

image = load_image(opt.image).resize((width, height))
mask_image = load_image(opt.mask).resize((width, height))
controlnet_conditioning_image = load_image(opt.hint).resize((width, height))

seed = opt.seed

if opt.prompt is not None and os.path.isfile(opt.prompt):
    print('reading prompts from prompt.txt')
    with open(opt.prompt, 'r') as f:
        prompt_from_file = f.readlines()
        prompt_from_file = [x.strip() for x in prompt_from_file if x.strip() != '']
        prompt_from_file = ', '.join(prompt_from_file)
        prompt = f'{prompt_from_file}, best quality, extremely detailed'
else:
    prompt = 'best quality, extremely detailed'

negative_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'

print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

for i in range(opt.n_samples):
    seed_i = seed + i * 1000
    generator = torch.manual_seed(seed_i)
    image = pipe(
        prompt,
        image,
        mask_image,
        controlnet_conditioning_image,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        width=width,
        height=height,
        generator=generator
    ).images[0]

    image.save(f"out_seed{seed_i}.png")