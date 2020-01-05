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

    #png.load()
    #background = Image.new("RGB", png.size, (255, 255, 255))
    #background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

    #gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray = lambda rgb : np.dot(rgb[... , :3] , [0.3333333 , 0.3333333, 0.3333333])
    #im = gray(np.asarray(background))
    im = gray(np.asarray(np.asarray(png)))

    return im

path = 'DSCN1784/'
	
paths = glob.glob(path + '*pos_z.png')
print(paths)

for im_path in paths:
    print(im_path)

    im = get_image(im_path)
    n, bins = np.histogram(im.ravel(),256,[0,256])

    #n, bins, ch = plt.hist(im.ravel(), 256, [0, 256])

    d = dict()

    x, y = im.shape
    max_1 = -1
    max_2 = -1
    v_1 = -1
    v_2 = -1

    for i in range(x):
        for j in range(y):
            v = int(round(im[i][j]))
            if v in d: 
                d[v] += 1
            else:
                d[v] = 1

            if d[v] > max_1:
                max_1 = d[v]
                v_1 = v
            elif d[v] > max_2:
                max_2 = d[v]
                v_2 = v

    if v_1 == 0.0:
        v_1 = v_2

    #argmax = np.argmax(n)
    threshold = 5

    #for i, j in d[v_1][1:]:
    #    im[i][j] = 0

    for i in range(x):
        for j in range(y):
            if int(im[i][j]) in range(v_1-threshold, v_1+threshold):
                im[i][j] = 0

    #a = np.asarray([i for i in im.ravel() if int(i) not in range(argmax-threshold, argmax+threshold)])

    plt.imshow(im)
    plt.axis('off')
    plt.savefig(im_path[:-4] + '_black.png',  bbox_inches='tight', dpi=600, pad_inches=0.0)
    #plt.show()
