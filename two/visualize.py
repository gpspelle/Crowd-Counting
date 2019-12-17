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

#paths = ['DSCN1650_pos.png', 'DSCN1666_pos.png']
paths = ['DSCN1810_pos.png']

for im_path in paths:

    im = get_image(im_path)
    n, bins = np.histogram(im.ravel(),256,[0,256])

    argmax = np.argmax(n)
    threshold = 5

    im_black = np.copy(im)
    x, y = im.shape

    for i in range(x):
        for j in range(y):
            if int(im[i][j]) in range(argmax-threshold, argmax+threshold):
                im_black[i][j] = 0

    #a = np.asarray([i for i in im.ravel() if int(i) not in range(argmax-threshold, argmax+threshold)])

    plt.imshow(im_black)
    plt.axis('off')
    plt.savefig(im_path[:-4] + '_black.png',  bbox_inches='tight', dpi=600, pad_inches=0.0)
