from PIL import Image
import imageio
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from statistics import stdev


def get_image(im_path):
    png = Image.open(im_path)

    png.load()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

    #gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray = lambda rgb : np.dot(rgb[... , :3] , [0.3333333 , 0.3333333, 0.3333333])
    im = gray(np.asarray(background))

    return im

im_path1 = 'DSCN1650_pos.png'
im_path2 = 'DSCN1666_pos.png'

im_1 = get_image(im_path1)
im_2 = get_image(im_path2)

plt.subplot(121)
plt.imshow(im_1, cmap='gray', vmin=0, vmax=255)


plt.subplot(122)
plt.imshow(im_2, cmap='gray', vmin=0, vmax=255)

plt.show()

plt.hist(im_1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.show()

