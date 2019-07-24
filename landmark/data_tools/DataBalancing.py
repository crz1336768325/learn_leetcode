import cv2
import numpy as np

import os
import glob
import platform
if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'
'''
dataset_path = [
 (os.path.join(os.getcwd(),'test1'),(os.path.join(os.getcwd(),'test1_balance1.txt'))),
                (os.path.join(os.getcwd(),'test2'),(os.path.join(os.getcwd(),'test2_balance2.txt'))),
                (os.path.join(os.getcwd(),'test3'),(os.path.join(os.getcwd(),'test3_balance3.txt'))),
                (os.path.join(os.getcwd(),'test4'),(os.path.join(os.getcwd(),'test4_balance4.txt'))),
                (os.path.join(os.getcwd(),'test5'),(os.path.join(os.getcwd(),'test5_balance5.txt'))),
		(os.path.join(os.getcwd(),'test6'),(os.path.join(os.getcwd(),'test6_balance6.txt'))),
                (os.path.join(os.getcwd(),'test7'),(os.path.join(os.getcwd(),'test7_balance7.txt'))),
                (os.path.join(os.getcwd(),'test8'),(os.path.join(os.getcwd(),'test8_balance8.txt'))),
                (os.path.join(os.getcwd(),'test9'),(os.path.join(os.getcwd(),'test9_balance9.txt'))),
                (os.path.join(os.getcwd(),'test10'),(os.path.join(os.getcwd(),'test10_balance10.txt'))),
		(os.path.join(os.getcwd(),'test11'),(os.path.join(os.getcwd(),'test11_balance11.txt')))
                ]
'''



dataset_path = [(os.path.join(os.getcwd(),'Face++_easy_1'),(os.path.join(os.getcwd(),'Face++_easy_1_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_2'),(os.path.join(os.getcwd(),'Face++_easy_2_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_3'),(os.path.join(os.getcwd(),'Face++_easy_3_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_4'),(os.path.join(os.getcwd(),'Face++_easy_4_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_5'),(os.path.join(os.getcwd(),'Face++_easy_5_wash.txt'))),
                (os.path.join(os.getcwd(),'Face++_easy_6'),(os.path.join(os.getcwd(),'Face++_easy_6_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_7'),(os.path.join(os.getcwd(),'Face++_easy_7_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_8'),(os.path.join(os.getcwd(),'Face++_easy_8_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_9'),(os.path.join(os.getcwd(),'Face++_easy_9_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_10'),(os.path.join(os.getcwd(),'Face++_easy_10_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_11'),(os.path.join(os.getcwd(),'Face++_easy_11_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_12'),(os.path.join(os.getcwd(),'Face++_easy_12_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_13'),(os.path.join(os.getcwd(),'Face++_easy_13_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_14'),(os.path.join(os.getcwd(),'Face++_easy_14_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_15'),(os.path.join(os.getcwd(),'Face++_easy_15_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_16'),(os.path.join(os.getcwd(),'Face++_easy_16_wash.txt'))),

(os.path.join(os.getcwd(),'Face++_easy_17'),(os.path.join(os.getcwd(),'Face++_easy_17_wash.txt'))),

(os.path.join(os.getcwd(),'Face++_easy_18'),(os.path.join(os.getcwd(),'Face++_easy_18_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_20'),(os.path.join(os.getcwd(),'Face++_easy_20_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_21'),(os.path.join(os.getcwd(),'Face++_easy_21_wash.txt'))),
(os.path.join(os.getcwd(),'Face++_easy_22'),(os.path.join(os.getcwd(),'Face++_easy_22_wash.txt')))
]




'''
dataset_path = [(os.path.join(os.getcwd(),'Face++_easy_9'),(os.path.join(os.getcwd(),'Face++_easy_9_wash.txt')))]
'''


'''
dataset_path = [(os.path.join(os.getcwd(),'Face++_easy_5'),(os.path.join(os.getcwd(),'Face++_easy_5_wash.txt')))]
'''
pose_dict = {'0':0,'1':0,'2':0,'3':0,'4':0}

print('count pose...')
for image_folder,txt_path in dataset_path:
    f = open(txt_path,'r')
    lines = f.readlines()
    
    for line in lines:
        line = line.strip().split(' ')
        try:
            image_path = os.path.join(image_folder,line[0])
            image = cv2.imread(image_path)
            h,w = image.shape[:2]
            
            pose = line[-1]
            pose_dict[pose] += 1
        except:
            continue
    
    f.close()

print('pose : {}'.format(pose_dict))#up_face:95 down_face:73 left_face:7988 right_face:5919 center:105843
# test1-6 rebalance pose : {'0': 95, '1': 73, '2': 7988, '3': 5919, '4': 52924}
# test1-9 rebalance pose : {'0': 179, '1': 89, '2': 11788, '3': 9223, '4': 60318}
# test1-10 rebalance pose : {'0': 262, '1': 117, '2': 14547, '3': 11810, '4': 66249}
## test1-11 rebalance pose : {'0': 305, '1': 132, '2': 16565, '3': 13502, '4': 70341}

#Face++_easy_1-11:pose : {'0': 305, '1': 132, '2': 16565, '3': 13502, '4': 140673}

#Face++_easy_1-15:pose : {'0': 305, '1': 132, '2': 16565, '3': 13502, '4': 140673}

