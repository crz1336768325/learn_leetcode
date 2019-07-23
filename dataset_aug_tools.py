import cv2
import numpy as np

import pickle
import shutil

import os
import glob
import platform
if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'

ImagesPath='dataset'

import numpy as np
import random
import cv2

def sp_noise(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
 
    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
 
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image

def Images2Pickles(mode='Train'):
    print('begin {} mode...'.format(mode))
    
    _dataset = 'dataset_AUG'
    f_txt = open(os.path.join(os.getcwd(),_dataset+'.txt'),'w')
    a=11
    for _image_path in glob.glob(os.path.join(os.getcwd(),ImagesPath,'*.jpg')):
        a+=1
        if a%1==0:       
            print('processing image {}'.format(_image_path))
            image_name = _image_path.split(SplitSym)[-1][:-4]
            print("imag_name",_image_path)
            BGR = cv2.imread(_image_path)

            image_name = '{}_{}.jpg'.format('dataset',a)
            image_name_flip='{}_{}.jpg'.format('dataset_flip',a)
            image_name_rotate='{}_{}.jpg'.format('dataset_rotate',a)
            image_name_gaussian='{}_{}.jpg'.format('dataset_gaussian',a)
            image_name_sp='{}_{}.jpg'.format('dataset_sp',a)
            line = '{}'.format(image_name)
            line = '{}\n'.format(line)

            line_flip = '{}'.format(image_name_flip)
            line_flip = '{}\n'.format(line_flip)

            line_rotate = '{}'.format(image_name_rotate)
            line_rotate = '{}\n'.format(line_rotate)

            line_gaussian = '{}'.format(image_name_gaussian)
            line_gaussian = '{}\n'.format(line_gaussian)

            line_sp = '{}'.format(image_name_sp)
            line_sp = '{}\n'.format(line_sp)



            f_txt.write(line)
            f_txt.write(line_flip)
            f_txt.write(line_rotate)
            f_txt.write(line_gaussian)
            f_txt.write(line_sp)


            path=os.path.join(os.getcwd(),_dataset,image_name)
            path_flip=os.path.join(os.getcwd(),_dataset,image_name_flip)
            path_rotate=os.path.join(os.getcwd(),_dataset,image_name_rotate)
            path_gaussian=os.path.join(os.getcwd(),_dataset,image_name_gaussian)
            path_sp=os.path.join(os.getcwd(),_dataset,image_name_sp)


            print("path",path)
            xImg = cv2.flip(BGR,1,dst=None) #flip the image
            rotateImg = cv2.rotate(BGR,0)  #rotate 90 
            gaussian_image=add_gaussian_noise(BGR,7)
            sp_image=sp_noise(BGR,0.02)
            cv2.imwrite(path,BGR)
            cv2.imwrite(path_flip,xImg)
            cv2.imwrite(path_rotate,rotateImg)
            cv2.imwrite(path_gaussian,gaussian_image)
            cv2.imwrite(path_sp,sp_image)

        
    f_txt.close()

if __name__=='__main__':
    Images2Pickles(mode='Train')
