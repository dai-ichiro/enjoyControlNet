import os
from PIL import Image
import numpy as np
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    '--image',
    required=True,
    type=str,
    help='original image'
)
args = parser.parse_args()

original_image = np.array(Image.open(args.image))

low_threshold_list = [50, 100, 150, 200, 250]

image_resolution = 512

apply_canny = CannyDetector()

img = resize_image(HWC3(original_image), image_resolution)
H, W, C = img.shape

for low_threshold in low_threshold_list:
    high_threshold_list = [x for x in low_threshold_list if x >= low_threshold]

    for high_threshold in high_threshold_list:
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        pil_image_detected_map = Image.fromarray(255 - detected_map)
        pil_image_detected_map.save(os.path.join('results', f'{low_threshold}_{high_threshold}.png'))

