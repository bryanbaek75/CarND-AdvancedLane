## Advanced Lane Finding   

### Self-driving car term1 p4, bryan baek. 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./output_images/undistortedimage4.jpg "Undistorted"
[image2]: ./output_images/undistortedimage.jpg "Road Transformed"
[image3]: ./output_images/pipelined.jpg "Pipelined"
[image4]: ./output_images/roadtransformed.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the calibrate function of line #23~#38 inside cam_calibrate.py. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color threshold and gradient thresholds to generate a binary image (thresholding steps at lines 56~80 in `tansform.py`).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `persTransform()`, which appears in lines 24 through 38 in the file `transform.py` .  The `persTransform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  The camera image is always 1280*720 in this case, I chose the hardcode the source and destination points in the following manner :

```python
src = np.float32(
      [[540,470],[760,470],[1260,720],[100,720]])
dst = np.float32(
      [[30,0],[1250,0],[1250,720],[30,720]]) 
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 540, 470  |    30, 0    |
| 100, 720  |   30, 720   |
| 1260, 720 |  1250, 720  |
| 760, 470  |   1250, 0   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with the centroid detection algorithm (find_window_centroid function, #79) , together with 2nd order polynomial fitting ( searchWindow function,#247~250), then applied average filtering ( filter function, #165 ) with sanity check ( sanityCheck function, #134) in findlane.py. 

The filter size is 5, but before adding current detected point to filter, it is sanity checked by the filled area size and distance from the old point location of previous time slot. 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated radius with filtered data in lines #261~#273 of my code in `findlane.py`

car center position is calculated in lines #32~#44 of my code in `lanefinder.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `lanefinder.py` in the function `findlane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Speed is import for the real environment, but  couldn't make fast enough without delay. So more improvement by utilizing previous detected area is required in future.  Also, the image pipelining parameter  is very sensitive to the brightness even thought the HLS color channel is applied. When the road color is changed, parameter must be self-adapted. The most big problem is that when the car is off the line once a lot, there will be many detection error and there is no lane marker on some road. So, real application, we need quite a lot of consideration of various situation and road environment. 

 