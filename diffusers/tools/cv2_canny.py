import os
from PIL import Image
import numpy as np
import cv2
from argparse import ArgumentParser

def canny_edge_detection(image:str, width:int=None, height:int=None) -> None:
    if width is None and height is None:
        original_image = np.array(Image.open(image))
    else:
        if width is None:
            original_image = np.array(Image.open(image).resize((height, height)))
        elif height is None:
            original_image = np.array(Image.open(image).resize((width, width)))
        else:
            original_image = np.array(Image.open(image).resize((width, height)))
        
    threshold1_list = [25, 50, 100, 150, 200]

    os.makedirs('canny_results', exist_ok=True)
    for threshold1 in threshold1_list:
        threshold2_list = [x for x in threshold1_list if x >= threshold1]
        for threshold2 in threshold2_list:
            control = cv2.Canny(original_image, threshold1=threshold1, threshold2=threshold2)
            Image.fromarray(control).save(os.path.join('canny_results', f'{threshold1}_{threshold2}.png'))

if __name__ == "__main__":
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
        help='width'
    )
    parser.add_argument(
        '--H',
        type=int,
        help='height'
    )
    args = parser.parse_args()
    canny_edge_detection(image=args.image, width=args.W, height=args.H)