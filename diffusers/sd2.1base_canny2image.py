from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
import os
import torch
from diffusers.utils import load_image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='original image'
)
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
parser.add_argument(
    '--n_samples',
    type=int,
    default=1,
    help='how many samples to produce for each given prompt',
)
parser.add_argument(
    '--model',
    type=str,
    default="model/stable-diffusion-2-1-base",
    help='path to model',
)
args = parser.parse_args()

model_id = args.model

n_samples = args.n_samples

if os.path.isdir(args.image):
    import glob
    hint_list = glob.glob(f'{args.image}/*.png')
    n_samples = 1
elif os.path.isfile(args.image):
    hint_list = [args.image]

vae_folder =args.vae
if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae').to('cuda')

controlnet=ControlNetModel.from_pretrained("controlnet/controlnet-sd21-canny-diffusers", torch_dtype=torch.float16)
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

print(f'n_samples: {n_samples}')
print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

os.makedirs('results', exist_ok=True)

seed = args.seed

for hint_image in hint_list:
    hint_fname = os.path.splitext(os.path.basename(hint_image))[0]
    canny_image = load_image(hint_image)
    for i in range(n_samples):
        seed_i = seed + i * 1000
        im = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt, 
            image=canny_image, 
            num_inference_steps=30,    
            generator=torch.manual_seed(seed_i)
        ).images[0]

        im.save(os.path.join('results', f'seed{seed_i}_{hint_fname}_result.png'))