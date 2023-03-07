import os
from PIL import Image
import numpy as np
import cv2
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--image',
    required=True,
    type=str,
    help='original image'
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

args = parser.parse_args()

width = args.W
height = args.H

original_image = np.array(Image.open(args.image).resize((width, height)))

threshold1_list = [50, 100, 150, 200]

os.makedirs('canny_results', exist_ok=True)
for threshold1 in threshold1_list:
    threshold2_list = [x for x in threshold1_list if x >= threshold1]
    for threshold2 in threshold2_list:
        control = cv2.Canny(original_image, threshold1=threshold1, threshold2=threshold2)
        Image.fromarray(control).save(os.path.join('canny_results', f'{threshold1}_{threshold2}.png'))