import keras
import params
import glob
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
import matplotlib.cm as cm
from skimage.measure import compare_ssim as ssim
import csv
import params

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def track(dir_1, dir_2, images_1, images_2, t1, t2):

    input = params.input
    x1 = dir_1.replace(input, '')
    #x2 = dir_2.replace(input, '')
    #x1 = x1[:-1]
    #x2 = x2[:-1]

    csv_name = params.input + x1 + "match.csv"
    #file = open(params.input + x1 + "_" + x2 + ".csv", "w") 
    #file = open(params.input + x1 + "match.csv", "w") 
    '''
    path = params.input
    output = path

    dirs = [f for f in glob.glob(path + '/*/')]
    dirs.sort()
    images = []
    code_2 = []
    for i in range(len(dirs)):
        code_2.append([f for f in glob.glob(dirs[i] + '/*code_2.npy')])
        images.append([f for f in glob.glob(dirs[i] + '/*_ori_*.png')])
        images[-1].sort()
    code_2.sort()
    code_2 = [item for sublist in code_2 for item in sublist]
    '''
    c1 = []
    c2 = []
    t1 = np.load(t1)
    t2 = np.load(t2)
    for rect in t1:
        x1, y1, x2, y2, _ = rect
        m1 = int((x1 + x2)/2)
        m2 = int((y1 + y2)/2)

        c1.append([m1, m2])

    c1 = np.asarray(c1)

    for rect in t2:
        x1, y1, x2, y2, _ = rect
        m1 = int((x1 + x2)/2)
        m2 = int((y1 + y2)/2)

        c2.append([m1, m2])

    c2 = np.asarray(c2)

    dists = np.zeros(shape=(len(c1), len(c2)))
    for i in range(len(c1)):
        for j in range(len(c2)):
            dists[i][j] = 1/ (((c1[i][0] - c2[j][0]) ** 2 + (c1[i][1] - c2[j][1]) ** 2) ** 0.5)

    #images = [item for sublist in images for item in sublist]

    #model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None)
    #model = keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None)

    imgs = []
    pred_1 = []
    pred_2 = []
    #pca = PCA(n_components=10)
    for img in images_1:
        im = cv2.imread(img, 0)
        im = cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
        pred_1.append(im)

    for img in images_2:
        im = cv2.imread(img, 0)
        im = cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
        pred_2.append(im)

    pred_1 = np.asarray(pred_1)
    pred_2 = np.asarray(pred_2)

    #pred = model.predict(l1)
    #flat_pred = pred.reshape(pred.shape[0], -1)
    #pca.fit(flat_pred)
    #reduced_X = pca.transform(flat_pred)

    #pred_1 = flat_pred
    #pred_1 = reduced_X
    '''
    if counter == 0:
        color = 'red'
    else:
        color = 'blue'

    for i in range(len(reduced_X)):
        plt.scatter(reduced_X[i][0], reduced_X[i][1], marker='x', color=color)
        plt.text(reduced_X[i][0]+.03, reduced_X[i][1]+.03, i, fontsize=9)

    counter+=1
    '''

    d = np.zeros(shape=(len(pred_1), len(pred_2)))
    min_ = min(len(pred_1), len(pred_2))
    for i in range(len(pred_1)):
        for j in range(len(pred_2)):
            d[i][j], _ = ssim(pred_1[i], pred_2[j], full=True)
            #d[i][j] = mse(pred_1[i], pred_2[j])

            #diff = pred_1[i] - pred_2[j]
            #gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            #media = sum(sum(gray)) / (x * y) 
            #cv2.imshow("merda", pred_1[i] - pred_2[j])
            #cv2.waitKey(0)
            #print("Comparing ", i, "to", j, ": ", cosine_similarity(pred_1[i].reshape(1,-1), pred_2[j].reshape(1,-1)))
            #print("Comparing ", i, "to", j, ": ", euclidean(pred_1[i].reshape(-1,1), pred_2[j].reshape(-1,1)))
            #print("Comparing ", i, "to", j, ": ", gray.mean())
        
    #print(dists[0])
    #exit(1)
    d_ = np.multiply(d, dists) 
    #d = np.multiply(d, dists) 

    present = []
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = ['first', 'second', 'd', 'dd']
        writer.writerow(header)
        while min_ != 0:
            x, y = np.unravel_index(np.argmax(d_, axis=None), d_.shape)
            present.append(x)
            row = [str(x), str(y), str(d[x][y]), str(d_[x][y])]
            #file.write(str(x) + "," + str(y) + ',' + str(d_[x][y]) + ',' + str(d[x][y]) + '\n')
            writer.writerow(row)
            '''
            print(d[x][y])
            numpy_horizontal = np.hstack((pred_1[x], pred_2[y]))
            while True:
                cv2.imshow('match', numpy_horizontal)
                k = cv2.waitKey(33)
                if k==27:    # Esc key to stop
                    break
                elif k==-1:  # normally -1 returned,so don't print it
                    continue
                else:
                    print(k) # else print its value
            '''
            d_[x, :] = -1
            d_[:, y] = -1
            #d_ = np.delete(d_, x, axis=0)
            #d_ = np.delete(d_, y, axis=1)
            
            #d = np.delete(d, x, axis=0)
            #d = np.delete(d, y, axis=1)

            #pred_1 = np.delete(pred_1, x, axis=0)
            #pred_2 = np.delete(pred_2, y, axis=0)

            min_-=1

        if len(pred_1) > len(pred_2):
            t = [i for i in range(len(pred_1))]
            l = [i for i in t if i not in present]
            for i in l:
                row = [str(i), "-1", "-100000", "-100000"]
                writer.writerow(row)
                
    #print("Head ", i, "matches to head", j)

    #plt.show()

    #for data in pred:
        #print(np.asarray(data).shape)
        #exit(1)
        #pca.fit(data)
        #reduced_X = pca.transform(data)
        #print(reduced_X)

''' 

    Idea: pegar o valor de mse e posição da imagem pra usar um algoritmo de clusterizaçao
    ou outra coisa pra melhorar o resultado

'''



