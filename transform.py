''' 
Function library for 
 - Perspective Transform and pipeline. 
 
Code by Bryan Baek 

 '''


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

src = None
dst = None
transform = None 
detransform = None 
img_size = None

def getSrc():
    return src

def initTransform(img):
    global src,dst,transform,detransform,img_size
    if( src is None ):
        img_size = img.shape
        print("Perspective Transform Parameter Initialized with image size : {}.".format(img_size))
        src = np.float32(
             [[540,470],[760,470],[1260,720],[100,720]])
        dst = np.float32(
            [[30,0],[1250,0],[1250,720],[30,720]])            
        print("Warp src: {}.".format(src))
        print("Warp dst: {}.".format(dst))                                    
    if(transform is None ):
        transform = cv2.getPerspectiveTransform(src,dst)        
    if(detransform is None ):
        detransform = cv2.getPerspectiveTransform(dst,src)        
        
def persTransform(img):
    global src,dst,transform,img_size
    if( transform is None ):
        initTransform(img)
    wraped = cv2.warpPerspective(img,transform,(img_size[1],img_size[0]),flags=cv2.INTER_LINEAR)        
    return(wraped) 

def dePersTransform(img):
    global src,dst,detransform,img_size
    if( detransform is None ):
        initTransform(img)    
    wraped = cv2.warpPerspective(img,detransform,(img_size[1],img_size[0]),flags=cv2.INTER_LINEAR)        
    return(wraped)

def pipeline(img, s_thresh=(100, 255), sx_thresh=(40, 100)):    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient  
    sxbinary = np.zeros_like(l_channel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1    

    # Threshold s color channel -- OK 
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary ==1)]= 1
    return combined_binary
    



