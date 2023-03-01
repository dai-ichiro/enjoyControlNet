import controlnet_hinter
from diffusers.utils import load_image
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--image',
    required=True,
    type=str,
    help='original image'
)
args = parser.parse_args()
original_image = np.array(load_image(args.image))
height, width, _ = original_image.shape

threshold1_list = [50, 100, 150, 200, 250, 300, 350, 400]

import os
os.makedirs('canny_results', exist_ok=True)

for threshold1 in threshold1_list:
    threshold2_list = [x for x in threshold1_list if x >= threshold1]
    for threshold2 in threshold2_list:
        control = controlnet_hinter.hint_canny(
            original_image, 
            low_threshold=threshold1, high_threshold=threshold2,
            width=width, height=height)
        
        control.save(os.path.join('canny_results', f'{threshold1}_{threshold2}.png'))