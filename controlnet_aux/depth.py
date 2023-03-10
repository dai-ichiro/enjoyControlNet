from transformers import pipeline
from diffusers.utils import load_image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='path to original image'
)
opt = parser.parse_args()

depth_estimator = pipeline('depth-estimation')

image = load_image(opt.image)
image = depth_estimator(image)['depth']
image.save('depth.png')

