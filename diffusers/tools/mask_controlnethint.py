from PIL import Image
import os
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--hint',
    type=str,
    required=True,
    help='controlnet hint image'
)
parser.add_argument(
    '--mask',
    type=str,
    required=True,
    help='mask image'
)
opt = parser.parse_args()

if os.path.isdir(opt.hint):
    import glob
    hint_list = glob.glob(f'{opt.hint}/*.png')
elif os.path.isfile(opt.hint):
    hint_list = [opt.hint]

mask_array = np.array(Image.open(opt.mask))

os.makedirs('mask_hint', exist_ok=True)

for hint_image in hint_list:
    hint_fname = os.path.basename(hint_image)
    hint_array = np.array(Image.open(hint_image))
    masked_hint_array = hint_array * (mask_array == 255)
    masked_hint_pilimage = Image.fromarray(masked_hint_array, mode='L')
    masked_hint_pilimage.save(os.path.join('mask_hint', hint_fname))

