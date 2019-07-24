import cv2
import numpy as np

import os
import glob
import platform
if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'
dataset_path = [(os.path.join(os.getcwd(),'test15'),(os.path.join(os.getcwd(),'test15.txt'))),
                
                ]
txt_name = 'test15'
#txt_path = os.path.join(os.getcwd(),txt_name+'.txt')
balance_txt = txt_name+'_balance15'
balance_path = os.path.join(os.getcwd(),balance_txt+'.txt')


g = open(balance_path,'w')  
'''
dataset_path = [(os.path.join(os.getcwd(),'Face++_easy_1'),(os.path.join(os.getcwd(),'Face++_easy_1_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_2'),(os.path.join(os.getcwd(),'Face++_easy_2_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_3'),(os.path.join(os.getcwd(),'Face++_easy_3_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_4'),(os.path.join(os.getcwd(),'Face++_easy_4_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_5'),(os.path.join(os.getcwd(),'Face++_easy_5_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_6'),(os.path.join(os.getcwd(),'Face++_easy_6_wash.txt')))]
'''
'''
dataset_path = [(os.path.join(os.getcwd(),'Face++_easy_5'),(os.path.join(os.getcwd(),'Face++_easy_5_wash.txt')))]
'''
pose_dict = {'0':0,'1':0,'2':0,'3':0,'4':0}

print('count pose...')
for image_folder,txt_path in dataset_path:
    f = open(txt_path,'r')
    lines = f.readlines()
    center=0
    for line in lines:
        line = line.strip().split(' ')
        
        image_path = os.path.join(image_folder,line[0])
        #image = cv2.imread(image_path)
        #h,w = image.shape[:2]
        image_name = line[0]
        #print(line)
        #print(image_name)
        landmark = list(map(float,line[1:167]))
        new_line = '{}'.format(image_name)
        for _landmark in landmark:
            new_line = '{} {}'.format(new_line,_landmark)
        gender = line[168]
        gender=int(gender)
        smile = float(line[169])
        smile=int(smile)
        #glass = list(map(float,line[170]))
        glass=int(line[170])
        glass=int(glass)
        pose = line[-1]
        pose_int=int(pose)
        if pose_int==4:
            center+=1
        if center==2:
            center=0
            os.remove(image_path)
        else:
            new_line = '{} {} {} {} {}\n'.format(new_line,gender,smile,glass,pose_int)
            g.write(new_line)
            
        
        #new_line = '{} {}'.format(new_line,landmark)
        
        



        pose_dict[pose] += 1

    
    f.close()
    g.close()
print('pose : {}'.format(pose_dict))#up_face:95 down_face:73 left_face:7988 right_face:5919 center:105843
