from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
import os
import torch
from diffusers.utils import load_image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed',
    type=int,
    default=20000,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--prompt',
    type=str,
    help='prompt file name'
)
parser.add_argument(
    '--vae',
    type=str,
    help='vae'
)
parser.add_argument(
    '--scheduler',
    type=str,
    default='eulera',
    choices=['pndm', 'multistepdpm', 'eulera']
)
args = parser.parse_args()

model_id = "model/waifu-diffusion"

vae_folder =args.vae
if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae').to('cuda')

controlnet=ControlNetModel.from_pretrained("controlnet/sd21-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    torch_dtype=torch.float16,
    controlnet=controlnet
).to('cuda')
scheduler = args.scheduler
match scheduler:
    case 'pmdn':
        from diffusers import  PNDMScheduler
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    case 'multistepdpm':
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    case 'eulera':
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    case _:
        None
pipe.enable_xformers_memory_efficient_attention()

canny_image = load_image('canny_results/100_200.png')

if args.prompt is not None and os.path.isfile(args.prompt):
    print(f'reading prompts from {args.prompt}')
    with open(args.prompt, 'r') as f:
        prompt_from_file = f.readlines()
        prompt_from_file = [x.strip() for x in prompt_from_file if x.strip() != '']
        prompt_from_file = ', '.join(prompt_from_file)
        prompt = f'{prompt_from_file}, best quality, extremely detailed'
else:
    prompt = 'best quality, extremely detailed'

negative_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'

print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

im = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt, 
    image=canny_image, 
    num_inference_steps=30,    
    generator=torch.manual_seed(args.seed)
).images[0]

im.save('canny_result.png')