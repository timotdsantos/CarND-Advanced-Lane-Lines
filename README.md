## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goal of this project is to write a software pipeline to identify the lane boundaries in a video. 

---

**Advanced Lane Finding Project**

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/distortion_correction_output.jpg "Undistorted"
[image2]: ./output_images/distortion_correction_output_lane.jpg "Road Transformed"
[image3]: ./output_images/binary_hls.png "HLS - S Channel"
[image4]: ./output_images/binary_hsv.png "HSV - V Channel"
[image_luv]: ./output_images/binary_luv.png "LUV - V Channel"
[image_lab]: ./output_images/binary_lab.png "LAB - B Channel"
[image_pipeline]: ./output_images/binary_pipeline.png "Pipeline Output"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup 

#### 1. This Readme is a detailed writeup of the project.  The project notebook which contains the code may be found here [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md), and will be referenced in the discussion.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in  cell [3] and [4] of the ipython notebook. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. For the calibration images, we're expecting a 9x6 corners. This is indicated by **(nx,ny)**, which is passed to `cv2.findChessboardCorners` which indicate how many corner points for the function to look for.


The `objpoints` will be appended with a copy of the (x,y,z) of each corner every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients **(dist** and **mtx)** using the `cv2.calibrateCamera()` function. The final calibration and distortion coefficients used for the entire project is calculated in [5], which uses the calibration chessboard foud in the repository (calibration2.jpg).

The camera calibration and distortion coefficients **(dist** and **mtx)** are used to un-distort the calibration image using `cv2.undistort()` function which can be seen under the function `cal_undistort()`. It can be seen in the undistorted image that the processing worked, as observed in the lack of a curvature in the top edge of the chessboard:

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The function `cal_undistort()` was used to undistort the example image. The mtx and dist computed from the calibration image is computed once used in undistorting the following image, and the rest of the images in the project. It's more obvious in the calibration image, but the undistorted test image below shows a slight different curvature in the hood portion ![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The code cells from [7] to [16] show various color space and thresholding steps I explored.

![alt text][image3]

![alt text][image4]

![alt text][image_lab]

![alt text][image_luv]


Finally, the pipeline I used for creating the binary image is composed of the following:
- **Gaussian Blur**
- **HLS, S-Channel**  -  It's good to note that it is able to capture the lane lines despite the road having different shade and color.
- **HSV, V-Channel**  -  It performs well at capturing the yellow line.
- **Sobel X** - performs the derivative on the x-axis to accentuate lines away from horizontal, this is effective in removing non-vertical lines like cars and shadows

Here is the output of the pipeline where each color is the contribution of the various color transform layers.
![alt text][image_pipeline]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
