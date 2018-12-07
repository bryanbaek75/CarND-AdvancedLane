''' 
Camera calibration &* distortion management library
written by bryan baek 
'''
import pickle 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import glob

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

objpoints = []

#object points 
nx = 9
ny = 6 

def calibrate():
    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        ret,corners,gray = findCorners(img)
        if( ret == True ):
            objpoints.append(objp)
            imgpoints.append(corners)
            print('Corner found for {} - {}'.format(fname,corners.shape))
            cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
            # cv2.imshow('Corner founds',img)
            # cv2.waitKey(100)
        else:
            print('Corner not found for {}.'.format(fname))
    return objpoints,imgpoints   

#image as input 
def findCorners(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)
    return ret,corners,gray 

def prepareUndistort(img,objpoints,imgpoints):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
    return mtx,dist 

def undistort(img,mtx,dist):
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    return undist 

def testUndistort():
    img = cv2.imread('./camera_cal/calibration4.jpg')    
    mtx,dist = prepareUndistort(img,objpoints,imgpoints)    
    undist = undistort(img,mtx,dist)
    cv2.imshow('Original Image',img)
    cv2.imshow('Undistorted Image',undist)
    cv2.imwrite("./output_images/undistortedimage4.jpg",undist)
    
    cv2.waitKey(0)

# calibrate()
# testUndistort() 
# cv2.destroyAllWindows()
