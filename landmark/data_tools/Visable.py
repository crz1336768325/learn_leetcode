import cv2
import numpy as np

import os
import glob
import platform
if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'

name = 'Face++_easy_300w'
images_path = name
target_txt_path = name+'_wash.txt'

g = open(target_txt_path,'r')
g_lines = g.readlines()

for i,line in enumerate(g_lines):
    line = line.strip().split(' ')
    image_name = line[0]
    image = cv2.imread(os.path.join(images_path,image_name))
    h,w = image.shape[:2]

    landmarks = list(map(float,line[1:167]))
    for j in range(0,len(landmarks),2):
        if j==102 or j==114:
            cv2.circle(image,(int(landmarks[j]),int(landmarks[j+1])),2,(0,255,0),-1)
        #cv2.circle(image,(int(landmarks[j]),int(landmarks[j+1])),2,(0,255,0),-1)
    cv2.imshow('test',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

g.close()
