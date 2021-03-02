import numpy as np
from PIL import Image
import random
from scipy import misc
import cv2
img=Image.open("qun.png")
img=np.array(img,dtype='float32')
img=img/255.

img=img[:,:,(0,1,2)]

img_size=int(img.size/3)

img1=img.reshape(img_size,3)

img1=np.transpose(img1)

img_cov=np.cov([img1[0],img1[1],img1[2]])

lamda,p=np.linalg.eig(img_cov)

p=np.transpose(p)
alpha1=random.normalvariate(0,0.3)
alpha2=random.normalvariate(0,0.3)
alpha3=random.normalvariate(0,0.3)
v=np.transpose((alpha1*lamda[0],alpha2*lamda[1],alpha3*lamda[2]))
add_num=np.dot(p,v)
print("add_num",add_num)
print("img[:,:,0]",img[:,:,0])
img2=np.array([img[:,:,0]+add_num[0],img[:,:,1]+add_num[1],img[:,:,2]+add_num[2]])
img2=np.swapaxes(img2,0,2)
img2=np.swapaxes(img2,0,1)
# misc.imsave("test.jpg",img2)

img2=img2/np.max(img2)*255
cv2.imwrite("test2.jpg",img2)