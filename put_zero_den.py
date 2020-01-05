import cv2
import numpy as np
import os
import glob
from paint import paint

path = 'DSCN1775/'
	
img = glob.glob(path + '*pos.png')

for img_path in img:
    paint(img_path,46,1703,391,167,591,140,917,1563)
	


