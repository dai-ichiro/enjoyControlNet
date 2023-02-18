from share import *

import einops
import numpy as np
import torch

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
import os
import sys
from PIL import Image

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--image',
    required=True,
    type=str,
    help='original image'
)
parser.add_argument(
    '--seed',
    type=int,
    default=200000,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=5,
    help='number of samples',
)
parser.add_argument(
    '--steps',
    type=int,
    default=20,
    help='steps',
)
parser.add_argument(
    '--scale',
    type=float,
    default=9.0,
    help='Guidance Scale',
)
parser.add_argument(
    '--low_threshold',
    type=int,
    default=100,
    help='Canny low threshold(min=1, max=255)',
)
parser.add_argument(
    '--high_threshold',
    type=int,
    default=200,
    help='Canny high threshold(min=1, max=255)',
)
args = parser.parse_args()

original_image = np.array(Image.open(args.image))
seed = args.seed
n_samples = args.n_samples
steps = args.steps
scale = args.scale
low_threshold = args.low_threshold
high_threshold = args.high_threshold

image_resolution = 512
eta = 0.0

if os.path.isfile('prompt.txt'):
    print('reading prompts from prompt.txt')
    with open('prompt.txt', 'r') as f:
        prompt = f.readlines()
        prompt = [x.strip() for x in prompt if x.strip() != '']
        prompt = ','.join(prompt)
else:
    print('Unable to find prompt.txt')
    sys.exit()

apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

def main():
    num_samples = 1
    with torch.no_grad():
        img = resize_image(HWC3(original_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        pil_image_detected_map = Image.fromarray(255 - detected_map)
        pil_image_detected_map.save(os.path.join('results', f'{low_threshold}_{high_threshold}.png'))

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)


        for i in range(n_samples):
            temp_seed = seed + i * 1000
            seed_everything(temp_seed)
            samples, intermediates = ddim_sampler.sample(steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = Image.fromarray(x_samples[0])
            results.save(os.path.join('results', f'result_seed{temp_seed}.png'))

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main()