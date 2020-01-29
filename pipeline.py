import os
import params

# Take photo

# Cut and paint in green

# Get density map with predictions
#os.system("python3 test.py")

# Put zero in non-queue area
#os.system("python3 put_zero_den.py")

# Remove noisy values
#os.system("python3 find_people.py")

# Get head's centroids and eliminate some false positives
#os.system("python3 position.py")

# Crop area
#os.system("python3 crop.py")

# Put zero in non-queue area
#os.system("python3 put_zero_image.py")

# Spot possible heads
#os.system("python3 tiny_face_eval.py --weight_file_path  weight --prob_thresh 0.04 --nms_thresh 0.0")

# Put heads into file 
#os.system("python3 get_heads.py")

# Track heads among videos
os.system("python3 track_video.py")

# Estimate velocity for every centroid paired between two images

# Remove centroids moving too fast

# Count number of people and track while they still inside the video

# With the records of people time in line: regression over it to estimate queue time from parameters like day of the week, time and number of people
