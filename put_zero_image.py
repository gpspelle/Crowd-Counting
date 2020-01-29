import cv2
import numpy as np
import os
import glob
from paint_image import paint
import params

path = params.input
	
dirs = [f for f in glob.glob(path + '/*/')]
images = []
for x in dirs:
    images.append([f for f in glob.glob(x + '/*_crop.png')])
images.sort()
images = [item for sublist in images for item in sublist]

for img_path in images:
    #paint(img_path, 36, 1785, 393, 75, 567, 60, 951, 1776)
    paint(img_path, 0, 3234, 737, 198, 1034, 220, 1617, 3228)
	


