from ldm.modules.structure_condition.model_edge import pidinet
from ldm.util import resize_numpy_image
import cv2
from basicsr.utils import img2tensor
import torch
import numpy as np
from PIL import Image
import argparse
import os 

os.makedirs('canny_results', exist_ok=True)

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
opt = parser.parse_args()

path_cond = opt.image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
# edge_generator
net_G = pidinet()
ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
net_G.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()}, strict=True)
net_G.to(device)

max_resolution = 512 * 512

im = cv2.imread(path_cond)
im = resize_numpy_image(im, max_resolution=max_resolution)
im = img2tensor(im).unsqueeze(0) / 255.

edge = net_G(im.to(device))[-1]

for threshold in opt.threshold_list:
    temp_edge = edge > threshold
    result_array = np.where(temp_edge.cpu().numpy() == False, 0, 255).squeeze().squeeze().astype(np.uint8)

    pil = Image.fromarray(result_array, mode='L')
    pil.save(os.path.join('canny_results', f'canny_{threshold}.png'))

