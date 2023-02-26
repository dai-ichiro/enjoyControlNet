import os 
os.makedirs('results', exist_ok=True)

import numpy as np
from diffusers import StableDiffusionControlNetPipeline, AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
import controlnet_hinter

import argparse
parser = argparse.ArgumentParser()
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
args = parser.parse_args()

steps = args.steps
threshold1 = args.threshold1
threshold2 = args.threshold2

if args.from_canny:
    control = load_image(args.image)
else:
    control = controlnet_hinter.hint_canny(np.array(load_image(args.image)), low_threshold=threshold1, high_threshold=threshold2)
    control.save(os.path.join('results', 'canny.png'))

base_model_id = 'Counterfeit-V2.5'  # an example: openjourney model
vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae').to('cuda')
unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder='unet').to('cuda')

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    'basemodel/control_sd15_canny',
    unet=unet,
    vae=vae,
    safety_checker=None).to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image = pipe(
    prompt='best quality, extremely detailed',  
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    image=control,
    width=512,
    height=512,
    num_inference_steps=steps).images[0]

image.save(os.path.join('results', 'generated.png'))
