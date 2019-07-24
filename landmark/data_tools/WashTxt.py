import numpy as np

import os
import glob
import platform
if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'

txt_name = 'Face++_easy_300w'
images_path = os.path.join(os.getcwd(),txt_name,'*.jpg')
txt_path = os.path.join(os.getcwd(),txt_name+'.txt')

wash_txt = txt_name+'_wash'
wash_txt_path = os.path.join(os.getcwd(),wash_txt+'.txt')

f = open(txt_path,'r')
g = open(wash_txt_path,'w')

for line in f.readlines():
    line = line.strip().split(' ')
    
    image_name = line[0]
    print(image_name)
    landmark = list(map(float,line[1:167]))
    emotion = list(map(float,line[167:174]))
    gender = line[174]
    age = float(line[175])
    left_eye_status = list(map(float,line[176:182]))
    right_eye_status = list(map(float,line[182:188]))
    headpose = list(map(float,line[188:191]))
    smile = float(line[191])
    
    new_line = '{}'.format(image_name)
    #81 landmark
    for _landmark in landmark:
        new_line = '{} {}'.format(new_line,_landmark)
    #gender 1(Female) 0(Male)
    if gender=='Male':
        gender = 0
    elif gender=='Female':
        gender = 1
    else:
        raise ValueError('gender must be Male or Female')
    #smile 1(>50,yes) 0(<=50,no)
    if smile>50:
        smile = 1
    else:
        smile = 0
    #glass 1(yes) 0(no)
    if (np.argmax(left_eye_status) in [0,4,5]) or (np.argmax(right_eye_status) in [0,4,5]):
        glass = 1
    else:
        glass = 0
    #headpose 0(pitch>30,up_face) 1(pitch<-30,down_face) 
    #         2(yaw>30,left_face) 3(yaw<-30,right_face) 
    #         4(-30<=yaw<=30 or -30<=pitch<=30,center)
    if headpose[1]>30:
        headpose = 0
    elif headpose[1]<-30:
        headpose = 1
    else:
        if headpose[0]>30:
            headpose = 2
        elif headpose[0]<-30:
            headpose = 3
        else:
            headpose = 4
    
    new_line = '{} {} {} {} {}\n'.format(new_line,gender,smile,glass,headpose)
    g.write(new_line)
    
    print('wash data of gender:{} smile:{} glass:{} headpose:{}'.format(gender,smile,glass,headpose))
    
f.close()
g.close()
