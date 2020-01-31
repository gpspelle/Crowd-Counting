import os

# Crop area
#os.system("python3 crop.py")


## APPROACH 1 MCNN

os.system("python3 put_zero_image.py")

os.system("python3 test.py")

os.system("python3 put_zero_den.py")

os.system("python3 find_people.py")

os.system("python3 position.py")


## APPROACH 2 - RNN

#os.system("python3 tiny_face_eval.py --weight_file_path  weight --prob_thresh 0.04 --nms_thresh 0.0")


## TRACKING 

# Put heads into file 
#os.system("python3 get_heads.py")

# Track heads among videos
#os.system("python3 track_video.py")
