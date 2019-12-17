from PIL import Image
import imageio
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from statistics import stdev

path = 'two/'
images = [f for f in glob.glob(path + "*pos_black.png")]
images.sort()
file = open("output_cluster.csv", "w")

for im_path in images:
    print(im_path)
    png = Image.open(im_path)

    png.load()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

    gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])

    im = gray(np.asarray(background))
    pos = np.where(im>110)
    #plt.plot(im)
    #plt.show()


    #print(pos)
    cart_pos = [[i[1], i[0]] for i in zip(pos[0], pos[1])]
    #print(cart_pos)

    clustering = DBSCAN(eps=13).fit(cart_pos)

    indices = [i for i, x in enumerate(clustering.labels_) if x == -1]

    labels_list = list(clustering.labels_)

    for index in sorted(indices, reverse=True):
        del labels_list[index]
        del cart_pos[index]

    number_cluster = len(set(labels_list))

    centroids = np.zeros(shape=(number_cluster, 2))

    for i in range(len(cart_pos)):
        centroids[clustering.labels_[i]][0] += cart_pos[i][0]
        centroids[clustering.labels_[i]][1] += cart_pos[i][1]

    #print(clustering.labels_)
    print(number_cluster)

    for i in range(number_cluster):
        num = list(clustering.labels_).count(i)
        #print(num, i)
        #print("Number of ", i, num)
        centroids[i] = centroids[i] // num

    #print("****************Centroids of clusters")
    #print(centroids)
    #print("****************Number of clusters")
    #print(number_cluster)
    #print("****************Core samples indices")
    #print(clustering.core_sample_indices_)

    silh_size = 50
    
    for i in range(number_cluster):
        if centroids[i][1] + silh_size >= im.shape[0]:
            centroids[i][0] = -1
            centroids[i][1] = -1

    silhuetas = np.zeros(shape=(number_cluster,silh_size))

    for i in range(number_cluster):
        for j in range(silh_size):
            if centroids[i][0] != -1:
                #print(int(centroids[i][0]), j + int(centroids[i][1]))
                silhuetas[i][j] = im[j + int(centroids[i][1])][int(centroids[i][0])]
            else:
                silhuetas[i][j] = -1

    #print(silhuetas)

    desvpads = np.zeros(number_cluster)
    for i in range(number_cluster):
        desvpads[i] = stdev(silhuetas[i])

    indices = [i for i, x in enumerate(desvpads) if x == 0]
    #print(indices)

    centroids = list(centroids)
    silhuetas = list(silhuetas)
    desvpads = list(desvpads)

    for index in sorted(indices, reverse=True):
        del centroids[index]
        del silhuetas[index]
        del desvpads[index]
    
    #print(silhuetas)
    #print(centroids)

    centroids = np.asarray(centroids)
    desvpads = np.asarray(desvpads)

    print(desvpads.reshape(-1,1))
    max = 0
    ind = 0

    #plt.imshow(im, cmap='gray', vmin=0, vmax=255)

    for i in range(centroids.shape[0]):
        plt.text(centroids[i][0], centroids[i][1], i, ha="center", va="center", fontdict={'color':'red', 'size':5})
    plt.close()
    #plt.show()
    if desvpads.size != 0:
        clustering_ = DBSCAN(eps=5, min_samples=1).fit(desvpads.reshape(-1,1))

        number_cluster_desvpad = len(set(clustering_.labels_))

        lista = list(clustering_.labels_)
        amount_per_cluster = []
        for i in range(number_cluster_desvpad):
            amount_per_cluster.append(lista.count(i))

        max = -1
        ind = -1
        for i in range(len(amount_per_cluster)):
            if amount_per_cluster[i] > max:
                max = amount_per_cluster[i]
                ind = i

        indices = [i for i, x in enumerate(lista) if x != ind]

        centroids = list(centroids)
        silhuetas = list(silhuetas)
        desvpads = list(desvpads)

        print(desvpads)
        print(indices)
        for index in sorted(indices, reverse=True):
            del centroids[index]
            del silhuetas[index]
            del desvpads[index]

        print(desvpads)
    
        centroids = np.array(centroids)
        print(centroids.shape)

        # put a red dot, size 40, at 2 locations:

        plt.imshow(im, cmap='gray', vmin=0, vmax=255)
        for i in range(centroids.shape[0]):
            plt.text(centroids[i][0], centroids[i][1], i, ha="center", va="center", fontdict={'color':'red', 'size':5})
        #plt.scatter(centroids[:,0], centroids[:,1], c='r', s=20)

        plt.axis('off')
        plt.savefig(im_path[:-4] + '_boxes.png',  bbox_inches='tight', dpi=600, pad_inches=0.0)

        plt.close()
        #plt.show()

        #print(len(indices))

    print(max, ind)
    file.write(im_path + "," + str(max) + '\n')
    
    


    
