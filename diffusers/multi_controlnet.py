from stable_diffusion_multi_controlnet import StableDiffusionMultiControlNetPipeline
from stable_diffusion_multi_controlnet import ControlNetProcessor
from diffusers import ControlNetModel, AutoencoderKL, EulerAncestralDiscreteScheduler
import torch
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
args = parser.parse_args()

model_id = args.model
vae_folder =args.vae

controlnet1 = args.controlnet1
controlnet2 = args.controlnet2

image1 = args.controlnet1_image
image2 = args.controlnet2_image

if vae_folder is not None:
    vae = AutoencoderKL.from_pretrained(vae_folder).to('cuda')
else:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae').to('cuda')

pipe = StableDiffusionMultiControlNetPipeline.from_pretrained(
    model_id,
    vae=vae,
    safety_checker=None).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

controlnet_processor1 = ControlNetModel.from_pretrained(controlnet1).to("cuda")
controlnet_processor2 = ControlNetModel.from_pretrained(controlnet2).to("cuda")

control_image1 = load_image(image1)  # load_image always return RGB format image
control_image2 = load_image(image2)  # refer to diffusers/src/diffusers/utils/testing_utils.py

prompt = "best quality, extremely detailed, cowboy shot"
negative_prompt = "cowboy, monochrome, lowres, bad anatomy, worst quality, low quality"
seed = 19

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    processors=[
        ControlNetProcessor(controlnet_processor2, control_image2),
        # ControlNetProcessor(controlnet_canny, canny_image),
    ],
    generator=torch.Generator(device="cpu").manual_seed(seed),
    num_inference_steps=30,
    width=512,
    height=512,
).images[0]
image.save(f"./mc_pose_only_result_{seed}.png")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    processors=[
        # ControlNetProcessor(controlnet_pose, pose_image),
        ControlNetProcessor(controlnet_processor1, control_image1),
    ],
    generator=torch.Generator(device="cpu").manual_seed(seed),
    num_inference_steps=30,
    width=512,
    height=512,
).images[0]
image.save(f"./mc_canny_only_result_{seed}.png")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    processors=[
        ControlNetProcessor(controlnet_processor2, control_image2),
        ControlNetProcessor(controlnet_processor1, control_image1),
    ],
    generator=torch.Generator(device="cpu").manual_seed(seed),
    num_inference_steps=30,
    width=512,
    height=512,
).images[0]
image.save(f"./mc_pose_and_canny_result_{seed}.png")