"""
Homework 5
Submission Functions
"""

# import packages here
import cv2 as cv
import numpy as np
#import scipy.optimize
import numpy.linalg as la
import matplotlib.pyplot as plt
import helper as hlp
from math import *



"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
#pts1 and pts2 are 2D arrays
def eight_point(pts1,pts2,M):
    #scaling the data
    N=len(pts1)
    h=np.ones((N,1),dtype='int')
    pts1=np.concatenate((pts1,h),axis=1) #making it a homogenous matrix Nx3
    pts2 =np.concatenate((pts2,h),axis=1)
    T=np.divide(np.eye(3),M) #transformation matrix
    T[-1,-1]=1

    pts1=np.dot(pts1,T)
    pts2=np.dot(pts2,T)

    A=[]
    for i,j in zip(pts1,pts2):
        x1,y1=i[0],i[1]
        x2,y2=j[0],j[1]
        li=[x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
        A.append(li)
    A=np.array(A)
    F=hlp._singularize(A)
    # refining(raising errors)
    #F = hlp.refineF(F, pts1, pts2)

    #unscaling f
    F=np.dot(np.transpose(T),np.dot(F,T))

    return F

"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""

def epipolar_correspondences(im1, im2, F, pts1):
    im1=cv.cvtColor(im1, cv.COLOR_BGR2GRAY).astype(np.float32)
    im2= cv.cvtColor(im2, cv.COLOR_BGR2GRAY).astype(np.float32)

    def create_window(image,center,w):
        x,y=center
        # Extract the region of interest from the original image
        window= image[y-w//2:y+w//2 ,x-w//2:x+w//2 ] #a window WITH WIDTH W
        return window

    def find_points_on_epipolar_line(point1,F,I2,w):
        # Define the search window around the epipolar line in the second image
        x1,y1 = point1
        sy, sx = I2.shape[:2]

        v = np.array([[x1], [y1], [1]])

        l = np.dot(F, v)
        s = np.sqrt(l[0, 0] ** 2 + l[1, 0] ** 2)

        if s == 0:
            print('Zero line vector in displayEpipolar')

        l = l / s

        if l[1, 0] != 0:
            xs = 0
            xe = sx
            ys = int(-(l[0, 0] * xs + l[2, 0]) / l[1, 0])
            ye = int(-(l[0, 0] * xe + l[2, 0]) / l[1, 0])
        else:
            ys = 0
            ye = sy
            xs = int(-(l[1, 0] * ys + l[2, 0]) / l[0, 0])
            xe = int(-(l[1, 0] * ye + l[2, 0]) / l[0, 0])

        # Collect the points lying on the epipolar line
        points_on_epipolar_line = []
        for x2 in range(xs+w//2,xe+1-w//2):
            y2 = int((-l[0] * x2 - l[2]) / l[1])
            for y2 in range(y2-1,y2+2):
               points_on_epipolar_line.append([x2,y2])
        candi_pts=np.array(points_on_epipolar_line)
        return(candi_pts)
    w=11
    pts2_lst=[]
    for i in pts1:

        window1=create_window(im1,i,w)
        l=np.dot(F,np.array([[i[0]],[i[1]],[1]])) #epipolar line
        s = sqrt(l[0,0] ** 2 + l[1,0] ** 2)

        if s == 0:
            print('Zero line vector')
        l = l/s

        min_cost=float('inf')
        corres_pt=None
        candi_pts=find_points_on_epipolar_line(i,F,im2,w) #GET THE CANDIDATE POINts
        # search for min intensity amongst the candi_pts
        for k in candi_pts:

            window2=create_window(im2,k,w)
            cost=sqrt(np.sum(np.square(np.subtract(window2,window1))))
            if cost<min_cost:
                min_cost=cost
                corres_pt=k
        pts2_lst.append(corres_pt)

    ptss2=np.array(pts2_lst)
    return ptss2
"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # E=k2(T)@F@K1
    E=np.dot(np.transpose(K2)@F,K1)
    return E

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    pass


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
