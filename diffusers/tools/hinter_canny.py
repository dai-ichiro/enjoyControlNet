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
parser.add_argument(
    '--resolution',
    type=int,
    help='resolution'
)

args = parser.parse_args()
original_image = np.array(load_image(args.image))
height, width, _ = original_image.shape

if height != width:
    print('need square image')
    import sys
    sys.exit()

if args.resolution is not None:
    resolution = args.resolution
else:
    resolution = width

threshold_list = [25, 50, 100, 150, 200, 250]

import os
os.makedirs('canny_results', exist_ok=True)

for threshold1 in threshold_list:
    new_list = [x for x in threshold_list if not x < threshold1]
    for threshold2 in new_list:
        control = controlnet_hinter.hint_canny(
            original_image, 
            low_threshold=threshold1, high_threshold=threshold2,
            width=resolution, height=resolution)
        
        control.save(os.path.join('canny_results', f'{threshold1}_{threshold2}.png'))