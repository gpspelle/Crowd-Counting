import numpy as np
import params
import glob
import cv2
from PIL import Image

path = params.input
output = path

dirs = [f for f in glob.glob(path + '/*/')]
dirs.sort()
code_2 = []
images = []
for x in dirs:
    code_2.append([f for f in glob.glob(x + '/*code_2.npy')])
    images.append([f for f in glob.glob(x + '/*_y.png')])
code_2.sort()
images.sort()
code_2 = [item for sublist in code_2 for item in sublist]
images = [item for sublist in images for item in sublist]

d = 32
dimx = 224
dimy = 224
for i in range(len(code_2)):
    t2 = np.load(code_2[i])
    img = cv2.imread(images[i])

    counter = 0
    for rect in t2:
        x1, y1, x2, y2, _ = rect

        m1 = int((x1 + x2)/2)
        m2 = int((y1 + y2)/2)
        
        v = img[m2-d:m2+d,m1-d:m1+d]
        v_ = cv2.resize(v, (dimy, dimx), interpolation=cv2.INTER_CUBIC)

        #cv2.imshow('bla', v_)
        #cv2.waitKey(0)

        cv2.imwrite(images[i][:-4] + '_ori_head_' + str(counter).zfill(4) + '.png', v) 
        cv2.imwrite(images[i][:-4] + '_res_head_' + str(counter).zfill(4) + '.png', v_) 

        #exit(1)
        #im_ = Image.fromarray(v_)
        #im_.save(images[i][:-4] + '_res_head_' + str(counter).zfill(4) + '.png')


        #im = Image.fromarray(v)
        #im.save(images[i][:-4] + '_ori_head_' + str(counter).zfill(4) + '.png')
        counter+=1
