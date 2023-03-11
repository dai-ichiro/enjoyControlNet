import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='model',
)
parser.add_argument(
    '--vae',
    type=str,
    help='vae'
)
parser.add_argument(
    '--controlnet1',
    type=str,
    required=True,
    help='controlnet1'
)
parser.add_argument(
    '--controlnet2',
    type=str,
    required=True,
    help='controlnet2'
)
parser.add_argument(
    '--controlnet1_image',
    type=str,
    required=True,
    help='image for controlnet1'
)
parser.add_argument(
    '--controlnet2_image',
    type=str,
    required=True,
    help='image for controlnet2'
)
parser.add_argument(
    '--seed',
    type=int,
    default=19,
    help='the seed (for reproducible sampling)',
)
args = parser.parse_args()

model_id = args.model
vae_folder =args.vae

controlnet1 = args.controlnet1
controlnet2 = args.controlnet2

image1 = args.controlnet1_image
image2 = args.controlnet2_image

if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder, torch_dtype=torch.float16).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae', torch_dtype=torch.float16).to('cuda')

controlnet_processor1 = ControlNetModel.from_pretrained(controlnet1, torch_dtype=torch.float16).to('cuda')
controlnet_processor2 = ControlNetModel.from_pretrained(controlnet2, torch_dtype=torch.float16).to('cuda')

'''
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    vae=vae,
    controlnet=[controlnet_processor1, controlnet_processor2],
    safety_checker=None,
    torch_dtype=torch.float16).to('cuda')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
'''
control_image1 = load_image(image1)  # load_image always return RGB format image
control_image2 = load_image(image2)  # refer to diffusers/src/diffusers/utils/testing_utils.py

prompt = "a beautiful girl wearing high neck sweater, best quality, extremely detailed, cowboy shot"
negative_prompt = "cowboy, monochrome, lowres, bad anatomy, worst quality, low quality"

seed = args.seed

'''
#controlnet1 only
pipe1 = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    vae=vae,
    controlnet=[controlnet_processor1],
    safety_checker=None,
    torch_dtype=torch.float16).to('cuda')
pipe1.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe1.scheduler.config)
pipe1.enable_xformers_memory_efficient_attention()

image = pipe1(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image = [control_image1],
    generator = torch.manual_seed(seed),
    num_inference_steps=30,
).images[0]
image.save(f"./controlnet1_only_result_{seed}.png")

#controlnet2 only
pipe2 = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    vae=vae,
    controlnet=[controlnet_processor2],
    safety_checker=None,
    torch_dtype=torch.float16).to('cuda')
pipe2.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe2.scheduler.config)
pipe2.enable_xformers_memory_efficient_attention()

image = pipe2(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image = [control_image2],
    generator = torch.manual_seed(seed),
    num_inference_steps=30,
).images[0]
image.save(f"./controlnet2_only_result_{seed}.png")
'''
#controlnet1 and controlnet2
pipe3 = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    vae=vae,
    controlnet=[controlnet_processor1, controlnet_processor2],
    safety_checker=None,
    torch_dtype=torch.float16).to('cuda')
pipe3.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe3.scheduler.config)
pipe3.enable_xformers_memory_efficient_attention()

image = pipe3(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image = [control_image1, control_image2],
    generator = torch.manual_seed(seed),
    num_inference_steps=30,
).images[0]
image.save(f"./controlnet_both_result_{seed}.png")#