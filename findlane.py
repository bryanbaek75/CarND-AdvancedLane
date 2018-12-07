''' 
Function library for lane detection
 - find centroids
 - 2 type of sanity check for  detected point in warped image and polyline 
 - filtering for smoothing  
 - drawing for information and debugging  

Code by Bryan Baek 

 '''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = []         
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
            


# Draw lane area with polyfit curve x,y 
def drawlane(warped):    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))     
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)        

#Original from the selfdriving car course 
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    # print("window_mask level,level*height,center,center.shape : {},{},{},{}".format(level,level*height,center,center.shape))
    # print("window_mask shape : {}".format(output.shape))
    return output

#Bryan Modified for point extraction 
def window_pointsmask(width, height, img_ref, center,level):
    points = numpy.zeros((center.shape[0],center.shape[0]))
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    # print("window_mask level,level*height,center : {},{},{}".format(level,level*height,center))
    # print("window_mask shape : {}".format(output.shape))
    return output

def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template    
    # Sum quarter bottom of image to get slice, could use a different ratio

    # print("find_window_centroids - image.shape = {}".format(image.shape))
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    # print("Centroid calculation. l_sum shape is {} and l_center is {} for {},{}".format(l_sum.shape,l_center,3*image.shape[0]/4,int(image.shape[1]/2)))    

    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    # print("first centroid - l_sum.shape={},r_sum.shape={},l_center={},r_center={}".format(l_sum.shape,r_sum.shape,l_center,r_center))
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    #Base for line detection. 
    left_base = 230 
    right_base = 1100
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)         
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window         
	    offset = window_width/2
	    l_min_index = int(max(left_base+offset-margin,0))
	    l_max_index = int(min(left_base+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset 
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(right_base+offset-margin,0))
	    r_max_index = int(min(right_base+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids    


def countFilled(warped,x,y,mask_size):    
    x_from = int(max(0, x-mask_size))    
    x_to = int(min(1280, x+mask_size)) 
    y_from = int(max(0,y-mask_size))
    y_to = int(min(720,y+mask_size))       
    # masksum = np.sum(warped[x_from:x_to,y_from:y_to])
    masksum = np.sum(warped[y_from:y_to,x_from:x_to])    
    # print("Cound from [{},{}] to [{}.{}] is {} for x={},y={},mask_size={} ".format(x_from,y_from,x_to,y_to,masksum, x,y,mask_size))
    return masksum

def sanityCheck(warped, points, lr="Left"):
    mask_size = 20 
    mask_thresh = 80 
    copy_warped = np.copy(warped) # copy for image pixeling.  
    # Sum of mark area
    # print("Warped image shape is {} and type is {}.".format(warped.shape,warped[0][0].dtype)) 
    # b_warped = np.empty((720,1280),boolean)
    b_warped = warped > 0 
    # print("Checking b_Warped image shape : {} and type : {}.".format(b_warped.shape,b_warped[0][0].dtype)) 
    # print("Testing bWarp sum is {}. if bigger the 10000, it is reasonable".format(np.sum(b_warped[0:720,0:1280])))        
    rpoints = []
    for i in range(0,len(points)):
        x = points[i][0]
        y = points[i][1]
        masksum = countFilled(b_warped,x,y,mask_size)    
        point = [x,y] 
        if(masksum >= mask_thresh):            
            rpoints.append(point)
            # print(" Point {} is passed & added with sum of {} with threshold {}.".format(point,masksum,mask_thresh))
        # else: 
            # print(" Point {} is faild with sum of {} with threshold {}.".format(point,masksum,mask_thresh)) 
        # cv2.circle(copy_warped,(int(x),int(y)), 20, (255,255,255), -1)              

    # cv2.imshow("Lane Finder SanityCheck "+lr,copy_warped)            
    # cv2.imwrite(".\output_images\LaneFinderSanityCheck-{}.jpg".format(lr),copy_warped)    
    return rpoints 


l_stack = []
r_stack = []
stack_size = 5
def filter(stack, ps, size):

    sanity_check_thresh = 900    
    if(len(stack)>0):         
        #sanity_check = np.sum(abs(ps - stack[len(stack)-1]))
        tmparray = np.asarray(stack)
        avgarray = tmparray.sum(0) / len(stack)
        sanity_check = np.sum(abs(ps - avgarray))
        # print("Point distance sanity = {}".format(sanity_check))        
        if(abs(sanity_check) > sanity_check_thresh):
            return(stack[len(stack)-1])

    stack.append(ps)
    if(len(stack)>size):
        stack.pop(0)  
    cnt = 0
    rps = np.zeros_like(ps)    
    for i in range(0,len(l_stack)):
        rps = rps + stack[i]
        cnt = cnt+1                 
    rps = rps / cnt     
    return(rps)    

#Global Variable for information image file save name change 
save_cnt = 1

#find center point of each centroid and return points 
def searchWindow(warped):     
    global l_stack,r_stack,stack_size,save_cnt
    # window settings
    window_width = 40 
    window_height = 30 # Break image into 9 vertical layers since image height is 720
    margin = 160 # How much to slide left and right for searching
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    rstr = ""
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros((len(window_centroids),2),dtype=np.int)
        r_points = np.zeros((len(window_centroids),2),dtype=np.int)
        # Go through each level and draw the windows 	

        # To draw all the left and right windows
        l_draw = np.zeros_like(warped)
        r_draw = np.zeros_like(warped)        

        for level in range(0,len(window_centroids)):
            l_points[level][0] = window_centroids[level][0]
            l_points[level][1] = warped.shape[0]- (window_height*level + window_height/2)
            r_points[level][0] = window_centroids[level][1]
            r_points[level][1] = warped.shape[0]- (window_height*level + window_height/2)    

            # # For information             
            # l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            # r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # l_draw[(l_draw == 255) | ((l_mask == 1) ) ] = 255
            # r_draw[(r_draw == 255) | ((r_mask == 1) ) ] = 255


        # print("L Points:{}".format(l_points[:,0]))        
        #check with mask, throw away if not appropriate. 
        l_points = sanityCheck(warped,l_points,"Left")
        r_points = sanityCheck(warped,r_points,"Right")

        # # Draw the results for information             
        # template = np.array(r_draw+l_draw,np.uint8) # add both left and right window pixels together
        # zero_channel = np.zeros_like(template) # create a zero color channel
        # template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        # warpage= np.dstack((warped, warped, warped))*128 # making the original road pixels 3 color channels
        # output = cv2.addWeighted(warpage, 1, template.astype(np.float64), 0.5, 0.0) # overlay the orignal road image with window results        
        # for i in range(0,len(l_points)):
        #     cv2.circle(output,(int(l_points[i][0]),int(l_points[i][1])), 10, (255,0,0),thickness=1)
        # for i in range(0,len(r_points)):
        #     cv2.circle(output,(int(r_points[i][0]),int(r_points[i][1])), 10, (255,0,0),thickness=1)            
        # cv2.imshow("LaneWindowing Sampleing",output)
        # cv2.imwrite(".\output_images\LaneWindowing Sampleing"+str(save_cnt)+".jpg",output)
        # save_cnt = save_cnt + 1 
        
        
        # polyfit
        ploty = np.linspace(0, 719, num=len(window_centroids)) # to cover same y-range as image
  
        left_fit = np.polyfit(np.asarray(l_points)[:,1], np.asarray(l_points)[:,0], 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit = np.polyfit(np.asarray(r_points)[:,1], np.asarray(r_points)[:,0], 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]                


        lps = np.column_stack((left_fitx,ploty))
        rps = np.column_stack((right_fitx,ploty))                

        lps = filter(l_stack,lps,stack_size)
        rps = filter(r_stack,rps,stack_size)


        # Cavature calculation after polyfit. 
        y_eval = np.max(ploty) 
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(np.asarray(lps)[:,1]*ym_per_pix, np.asarray(lps)[:,0]*xm_per_pix, 2)
        right_fit_cr = np.polyfit(np.asarray(rps)[:,1]*ym_per_pix, np.asarray(rps)[:,0]*xm_per_pix, 2)
        # Calculate the new radii of curvature in meters 
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])                
        last = len(lps)-1
        rstr = "Carvature left:{:0.2f}m, right:{:0.2f}m".format(left_curverad, right_curverad)
    # If no window centers found, just display orginal road image
    else:
        print("No Window found.")
    output = np.zeros_like(warped)        
    lps = np.int32(lps)
    rps = np.flip(np.int32(rps),0)    
    output = cv2.fillPoly(output,[np.concatenate((lps,rps), axis=0)],1)           
    return output,rstr

    