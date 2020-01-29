from PIL import Image
import imageio
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from statistics import stdev
import params


def get_image(im_path):
    png = Image.open(im_path)

    png.load()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

    #gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray = lambda rgb : np.dot(rgb[... , :3] , [0.3333333 , 0.3333333, 0.3333333])
    im = gray(np.asarray(background))

    return im

im_path1 = params.input + 'DSCN1653_pos_z_black.png'
#im_path2 = 'DSCN1666_pos_black.png'
#im_path3 = 'DSCN1810_pos_black.png'

im_1 = get_image(im_path1)
#im_2 = get_image(im_path2)
#im_3 = get_image(im_path3)

#plt.subplot(131)
plt.imshow(im_1, cmap='gray', vmin=0, vmax=255)

#plt.subplot(132)
#plt.imshow(im_2, cmap='gray', vmin=0, vmax=255)

#plt.subplot(133)
#plt.imshow(im_3, cmap='gray', vmin=0, vmax=255)
plt.show()


