import numpy as np
import cv2 as cv
import submission as sub
import helper as hlp
import matplotlib.pyplot as plt
from math import *

im1=cv.imread(r'data\im1.png')
im2=cv.imread(r'data\im2.png')
data=np.load("data\some_corresp.npz")
pts1=data['pts1']
pts2=data['pts2']
M=max(im1.shape[0],im1.shape[1])

#EIGHT POINT ALGORITHM
F=sub.eight_point(pts1,pts2,M)
print('Fundamental matrix:\n',F)
#hlp.displayEpipolarF(im1,im2, F)

#EPIPOLAR CORRESPONDANCES
templ=np.load(r"data\temple_coords.npz")
P1=templ['pts1']
P2=sub.epipolar_correspondences(im1,im2,F,P1)
hlp.epipolarMatchGUI(im1,im2,F)

#ESSENTIAL MATRIX
intrs=np.load(r"data\intrinsics.npz")
k1=intrs['K1']
k2=intrs['K2']
E=sub.essential_matrix(F,k1,k2)
print("Essential matrix:\n",E)