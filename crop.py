import cv2
import glob
import params

path = params.input
output = path

dirs = [f for f in glob.glob(path + '/*/')]
images = []
for x in dirs:
    images.append([f for f in glob.glob(x + '/*.JPG')])
images.sort()
images = [item for sublist in images for item in sublist]

for img_path in images:
    img = cv2.imread(img_path)
    print(img.shape)
    #exit(1)
    #img_crop = img[1288:2920,2512:]
    img_crop = img[2512:,1288:2920]
    cv2.imwrite (img_path[:-4] + '_crop.png', img_crop)
