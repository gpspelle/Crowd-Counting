import glob
import os

for file in glob.glob("*.JPG"):
    try:
        os.makedirs(file[:-4])
        os.rename(file, file[:-4] + '/' + file)

    except OSError:
        print ("Creation of the directory %s failed" % file[:-4])

