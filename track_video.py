import params
import glob
import numpy as np
import pandas as pd
import cv2
from track_heads import track
path = params.input
output = path

dirs = [f for f in glob.glob(path + '/*/')]
dirs.sort()
images = []
res_images = []
code_2 = []
for i in range(len(dirs)):
    code_2.append([f for f in glob.glob(dirs[i] + '/*code_2.npy')])
    images.append([f for f in glob.glob(dirs[i] + '/*_ori_*.png')])
    res_images.append([f for f in glob.glob(dirs[i] + '/*_res_*.png')])
    images[-1].sort()
    res_images[-1].sort()
code_2.sort()
code_2 = [item for sublist in code_2 for item in sublist]

#for i in range(len(dirs)-1):
#    print(images[i], images[i+1], code_2[i], code_2[i+1])
#    track(dirs[i], dirs[i+1], images[i], images[i+1], code_2[i], code_2[i+1])

#n_track = 10

dataset = pd.read_csv(dirs[0] + 'match.csv')
df = pd.DataFrame(dataset)
n_track = df.shape[0]
ids = np.zeros(shape=(n_track, len(dirs)-1, 2), dtype=np.int8)
dists = np.zeros(shape=(n_track, len(dirs)-1), dtype=np.float32)
#df = df[:n_track]
#print(df)
#for index, row in df.iterrows():
#    print(row['first'])
#exit(1)
for j, row in df.iterrows():
    #v1 = row.split(',')
    ids[j][0][0] = row['first'] 
    ids[j][0][1] = row['second'] 
    dists[j][0] = row['dd']
    
for i in range(1, len(dirs)-1):
    #print(i, end='')
    dataset = pd.read_csv(dirs[i] + 'match.csv')
    df = pd.DataFrame(dataset)
    for j in range(n_track):
        #print(" ", j, end='')
        search = ids[j][i-1][1]
        if search != -1:
            s = df.loc[df['first'] == search]
            v1 = s['second'].to_numpy()[0]
            dd = s['dd'].to_numpy()[0]

            ids[j][i][0] = int(search)
            ids[j][i][1] = int(v1)
            dists[j][i] = dd
        else:
            ids[j][i][0] = -1 
            ids[j][i][1] = -1
            dists[j][i] = np.random.rand()*100

    print(ids)

num_tracked = 0
list_p = []
people_pos = []
for j in range(n_track):
    #print("Tracked number ", j)
    stacks = []
    stack = []
    #if np.std(dists[j]) > 1:
    #    print("To passando pq deu merda, desvpad mt alto")
    #    continue

    size_hori = 3
    counter = 0
    resx = resy = 0
    l_p = []
    p_pos = []
    for i in range(len(dirs)-1):
        t1 = np.load(code_2[i])
        x1, y1, x2, y2, _ = t1[ids[j][i][0]]
        m1 = int((x1 + x2)/2)
        m2 = int((y1 + y2)/2)

        p_pos.append([m1, m2])


        im = cv2.imread(res_images[i][ids[j][i][0]])
        l_p.append(ids[j][i][0])
        resx, resy, _ = im.shape
        stack.append(im)
        #print(images[i][ids[j][i][0]])
        counter+=1
        if counter == size_hori:
            counter = 0
            stack = np.asarray(stack)
            stacks.append(np.hstack(stack))
            stack = []


    blank = np.zeros((resx, resy, 3), np.uint8)
    im = cv2.imread(res_images[len(dirs)-1][ids[j][len(dirs)-2][1]])
    l_p.append(ids[j][len(dirs)-2][1])
    stack.append(im)
    counter+=1

    while counter != size_hori:
        stack.append(blank)
        counter += 1

    stack = np.asarray(stack)
    stacks.append(np.hstack(stack))
    #stack = np.asarray(stack)
    #stacks.append(np.hstack(stack))
    vertical_stack = np.vstack(stacks)

    #print(dists[j])
    #print(ids[j])

    bosta = False
    #for b in dists[j]:
    #    if b < 0.001:
    #        bosta = True
    #        break

    #if bosta:
    #    print("To passando pq deu bosta, imagems nada a ver")
    #    continue

    num_tracked+=1

    list_p.append(l_p)
    people_pos.append(p_pos)
    #'''
    while True:
        cv2.imshow('match ' + np.array2string(dists[j]) + ' ' + str(np.std(dists[j])), vertical_stack)
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print(k) # else print its value
    #'''
    #print(images[len(dirs)-1][ids[j][len(dirs)-2][1]])
    #print("####################")
    #numpy_horizontal = np.hstack((pred_1[x], pred_2[y]))

print(num_tracked)
print(list_p)
print(people_pos)
np.save("kris", np.asarray(people_pos))

#print(ids)
