from controlnet_aux import HEDdetector
import numpy as np
from PIL import Image
import argparse
import os 

os.makedirs('sketch_results', exist_ok=True)

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
    default=512,
    help='resolution'
)
opt = parser.parse_args()

resolution = opt.resolution
image = Image.open(opt.image).resize((resolution, resolution))

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
hed_array = np.array(hed(image, detect_resolution=resolution, image_resolution=resolution).convert('L'))

for threshold in opt.threshold_list:
    bool_array = hed_array > (255 * threshold)
    result_array = np.where(bool_array == False, 0, 255).astype(np.uint8)

    pil = Image.fromarray(result_array, mode='L')
    pil.save(os.path.join('sketch_results', f'sketch_{threshold}.png'))

