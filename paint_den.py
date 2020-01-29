import cv2
import numpy as np
import os
import glob
import params

'''
    This code receives an image path and four coordinates in a cartesian plane
    and color the area not delimited by these four points with black.
'''


def paint(img_path,x1,y1,x2,y2,x3,y3,x4,y4):
    img = cv2.imread(img_path)
    linhas,colunas,layer = img.shape

    xres = params.original_xres
    yres = params.original_yres

    dx = colunas/xres
    dy = linhas/yres

    x1*=dx
    x2*=dx
    x3*=dx
    x4*=dx
    y1*=dy
    y2*=dy
    y3*=dy
    y4*=dy
    

    a1 = (y1-y2)/(x1-x2)
    b1 = y1 - a1*x1
    a2 = (y3-y4)/(x3-x4)
    b2 = y3 - a2*x3
    img = np.array(img)

    #print(a1, b1)
    #print(a2, b2)
    
    #x2 = (0 - b1) / a1
    #x3 = (0 - b2)/a2
    #x1 = (linhas-1 - b1) / a1
    #x4 = (linhas-1 - b2) / a2

    #print(x1, linhas-1)
    #print(x2, 0)
    #print(x3, 0)
    #print(x4, linhas-1)

    for y in range (linhas):
            
        posX1 = int((y-b1) / a1)
        posX2 = int((y - b2)/a2)

        img[y, :posX1] = [0, 0, 0]	
        img[y, posX2:] = [0, 0, 0]
	
    cv2.imwrite(img_path[:-4] + "_z.png",img)
