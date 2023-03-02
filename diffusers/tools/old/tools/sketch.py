import os
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    help='original image'
)
parser.add_argument(
    '--size',
    type=int,
    default=4,
    help='rectangle size'
)

def nothing(x):
    pass

opt = parser.parse_args()
size = opt.size

img_path = opt.image
img_fname_no_ext = os.path.splitext(os.path.basename(img_path))[0]

drawing = False 

def draw(event,x,y, flags, param):

    global drawing, mask_image

    s = cv2.getTrackbarPos('Size','image') * 3 + 3
    i = cv2.getTrackbarPos(switch,'image')

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing == True:
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
            xmin = x - s
            ymin = y - s
            xmax = x + s
            ymax = y + s

            if i == 0:
                cv2.rectangle(original_image, (xmin,ymin), (xmax, ymax), 255, -1)
            elif i == 1:
                cv2.rectangle(original_image, (xmin,ymin), (xmax, ymax), 0, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('image')
switch = 'draw / erase'
cv2.createTrackbar('Size', 'image', 0, 2,nothing)
cv2.createTrackbar(switch, 'image', 0, 1, nothing)
cv2.setMouseCallback('image', draw)

while True:  
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        cv2.imwrite(f'{img_fname_no_ext}_modified.png', original_image)
    elif k & 0xFF == 27:
        break
    cv2.imshow('image',original_image)

cv2.destroyAllWindows()