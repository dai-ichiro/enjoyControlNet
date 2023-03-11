from controlnet_aux import HEDdetector
import numpy as np
from PIL import Image
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--threshold_list',
    nargs='*',
    default=[0.5],    
    type=float,
    help='threshold',
)
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='path to original image'
)
parser.add_argument(
    '--resolution',
    type=int,
    help='resolution'
)
parser.add_argument(
    '--all',
    action='store_true',
    help='if true, threshold from 0.1 to 0.9'
)
opt = parser.parse_args()

resolution = opt.resolution

if resolution is None:
    image = Image.open(opt.image)
    resolution = image.height
else:
    image = Image.open(opt.image).resize((resolution, resolution))

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
hed_array = np.array(hed(image, detect_resolution=resolution, image_resolution=resolution).convert('L'))

if opt.all:
    threshold_list = [x * 0.1 for x in range(1, 10)]
else:
    threshold_list = opt.threshold

os.makedirs('scribble_results', exist_ok=True)

for threshold in threshold_list:
    bool_array = hed_array > (255 * threshold)
    result_array = np.where(bool_array == False, 0, 255).astype(np.uint8)

    pil = Image.fromarray(result_array, mode='L')
    pil.save(os.path.join('scribble_results', f'sketch{threshold:.1f}.png'))

