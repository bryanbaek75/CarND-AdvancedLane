''' 
Main python library for advanced lane finding
written by bryan baek 
'''

import numpy as np 
import cv2 
import cam_calibrate as cal 
import transform as tf
import findlane as fl  
import glob

INPUT_FILE = 'project_video.mp4'
OUTPUT_FILE = 'output.mp4'

cap = cv2.VideoCapture(INPUT_FILE)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_FILE,fourcc,20.0,(1280,720))
SAMPLE_FILENAME = './test/sample1.jpg'
base_img = None
mtx = None
dmtx = None
print("Current working CV2 version is {}.".format(cv2.__version__))

def initialize():
    global objpoints, imgpoints,mtx,dist 
    base_img = cv2.imread(SAMPLE_FILENAME)
    objpoints, imgpoints = cal.calibrate()
    mtx,dist = cal.prepareUndistort(base_img,objpoints,imgpoints)

# find car center from dewarped result data 
def findCarCenter(warp):    
    check_y = warp.shape[0]-1
    xmin = 100000
    xmax = 0 
    for i in range(0, warp.shape[1]):
        if(warp[check_y][i] > 0 ):
            xmin = min(i,xmin)
            xmax = max(i,xmax)
    # Car Location from center 
    base = (1280 - (xmax - xmin)) / 2  
    car_loc = ( xmin - base ) * 3.7 / (xmax - xmin)
    # print("Car Center find - shape {},xmin {},xmax {}".format(warp.shape,xmin,xmax))    
    return car_loc

def findLane(frame):
    # print("trying to find lane...")
    if(mtx is None):
        print("[Critical] Mtx is still none. ")    
    undist = cal.undistort(frame,mtx,dist)    
    warped = tf.persTransform(undist)    
    pipelined = tf.pipeline(warped)
    windowed,rstr = fl.searchWindow(pipelined)
    warp = tf.dePersTransform(windowed)
    rstr2 = " ,car from center = {:0.02f} m".format(findCarCenter(warp))
    rstr = rstr + rstr2 
    print(rstr)           
    newwarp = np.zeros(undist.shape,undist.dtype)
    newwarp[:,:,1] = warp[:,:]*255        
    result = cv2.addWeighted(undist,1, newwarp, 0.3, 0)
    #print curvature, location in image 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,rstr,(50,50), font, 1,(255,255,255),1,cv2.LINE_AA)
    return result

def testPerspective():
    images = glob.glob('./test/saved_screenshot*.jpg')
    for idx, fname in enumerate(images):    
        img = cv2.imread(fname)            
        warped = tf.persTransform(img)
        pipelined = tf.pipeline(warped)
        windowed,rstr = fl.searchWindow(pipelined)    
        warp = tf.dePersTransform(windowed)        
        newwarp = np.zeros(img.shape,img.dtype)        
        newwarp[:,:,1] = warp[:,:]*255        
        result = cv2.addWeighted(img,1, newwarp, 0.3, 0)
        print(tf.getSrc())
        poly = np.array(tf.getSrc(), dtype=np.int32 )
        print("Perspective poly is {} with shape {}, type {}.".format(poly,poly.shape,poly.dtype))
        result = cv2.polylines(result,[poly],1,(0,0,255))  
        cv2.imshow("Image",result)
        cv2.imwrite(fname.replace("saved_","pers_"),result)                    
        cv2.waitKey(0)


def testLaneFind():
    img = cv2.imread(SAMPLE_FILENAME)
    print("test image shape = {}, height ={}, width = {}".format(img.shape, img.shape[0],img.shape[1]))
    # cv2.imshow("Lane Finder Original.",img)        
    # cv2.imwrite(".\output_images\LaneFinderOriginal1.jpg",img)

    warped = findLane(img)
    # cv2.imshow("Lane Finder Test.",warped)    
    pipelined = tf.pipeline(warped)
    # cv2.imshow("Lane Finder pipelined.",pipelined)
    # cv2.imwrite(".\output_images\LaneFinderPipelined1.png",pipelined)    
    windowed = fl.searchWindow(pipelined)
    # cv2.imshow("Lane Finder Windowed.",windowed)
    # cv2.imwrite(".\output_images\LaneFinderWindowed1.jpg",windowed)    

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warp = tf.dePersTransform(windowed)
    newwarp = np.zeros(img.shape,img.dtype)
    newwarp[:,:,1] = warp[:,:]*255        
    print("Original image shape is {}, pipeline warped image shape is {}.".format(img.shape,newwarp.shape)) 
    # cv2.imshow("Lane Finder Warp Result",newwarp)     
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    cv2.imshow("Lane Finder Result",result)
    # cv2.imwrite(".\output_images\LaneFinderResult1.jpg",result)    

def testPipeline():
    img = cv2.imread(SAMPLE_FILENAME)
    print("test image shape = {}, height ={}, width = {}".format(img.shape, img.shape[0],img.shape[1]))
    cv2.imshow("Lane Finder Original.",img)        
    pipelined = tf.pipeline(img)    
    newpip = np.zeros(img.shape,img.dtype)
    newpip[:,:,0] = pipelined[:,:]*255                         
    newpip[:,:,1] = pipelined[:,:]*255                     
    newpip[:,:,2] = pipelined[:,:]*255                         
    cv2.imshow("Lane Finder pipelined.",newpip)
    cv2.imwrite("./output_images/pipelined.jpg",newpip)             

savecount = 1 
SAVE_FILENAME = './test/saved_screenshot'

def doLaneFind():         
    global savecount,SAVE_FILENAME 
    while(cap.isOpened()):
        ret,frame = cap.read()
        if(ret == True):        
            lane_frame = findLane(frame)
            cv2.imshow('Advanced Lane Finder',lane_frame)
            out.write(lane_frame)                    
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break
            else:
                if(cv2.waitKey(1) & 0xFF == ord('s')):
                    filename=SAVE_FILENAME+str(savecount)+"L.jpg"
                    cv2.imwrite(filename,lane_frame)                    
                    print("Image file [{}] saved.".format(filename))
                    savecount=savecount+1
        else:
            break        
    cap.release()
    out.release()
    print("Lane finding complete.")

initialize()
doLaneFind()
# testLaneFind()
# testPipeline()
# testPerspective()
cv2.waitKey(0)        
cv2.destroyAllWindows()





