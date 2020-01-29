import numpy as np
import params
import glob
import cv2

def calc_area(Ax, Ay, Bx, By, Cx, Cy):
    return np.absolute(Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By)) / 2

path = params.input
output = path

dirs = [f for f in glob.glob(path + '/*/')]
dirs.sort()
code_1 = []
code_2 = []
for x in dirs:
    code_1.append([f for f in glob.glob(x + '/*code_1.npy')])
    code_2.append([f for f in glob.glob(x + '/*code_2.npy')])
code_1.sort()
code_2.sort()
code_1 = [item for sublist in code_1 for item in sublist]
code_2 = [item for sublist in code_2 for item in sublist]

file1 = open("output_mix.csv", "w")
file2 = open("output_new.csv", "w")
for i in range(len(code_1)):
    counter = 0
    blank_image = np.zeros((params.original_yres,params.original_xres,3), np.uint8)
    rect_color = [255, 0, 0]
    d = 20
    t1 = np.load(code_1[i])
    t2 = np.load(code_2[i])
    for point in t1:

        Py, Px = point

        Px = int(Px)
        Py = int(Py)

        px1 = int(Px - d/2)
        px2 = int(Px + d/2)

        py1 = int(Py - d/2)
        py2 = int(Py + d/2)

        #print(px1, px2, py1, py2)

        cv2.rectangle(blank_image, (px1, py1), (px2, py2), rect_color, 1)
    for rect in t2:
        x1, y1, x2, y2, _ = rect

        x1 -= 10
        y1 -= 10

        x2 += 10
        y2 += 10
        Ax = x1
        Ay = y1

        Bx = x2
        By = y1

        Cx = x1
        Cy = y2

        Dx = x2
        Dy = y2

        area_q = abs(x2-x1) * abs(y2-y1)
        point_inside = False
        for point in t1:
            Py, Px = point

            Px = int(Px)
            Py = int(Py)

            # APD = ABC
            area_t = calc_area(Ax, Ay, Px, Py, Dx, Dy) 

            # DPC = ABC
            area_t += calc_area(Dx, Dy, Px, Py, Cx, Cy)

            # CPB = ABC
            area_t += calc_area(Cx, Cy, Px, Py, Bx, By)

            # PBA = ABC
            area_t += calc_area(Px, Py, Bx, By, Ax, Ay)


            if area_t <= area_q:
                point_inside = True
                break

        if not point_inside:
            counter += 1
            cv2.rectangle(blank_image, (x1, y1), (x2, y2), rect_color, 1)
        
    file1.write(dirs[i] + "," + str(len(t1) + counter) + '\n')
    file2.write(dirs[i] + "," + str(len(t2)) + '\n')
    #print(len(code_1[i]) + counter)
    cv2.imwrite("file.png", blank_image)
    cv2.imshow("image", blank_image)
    cv2.waitKey(0)

