## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goal of this project is to write a software pipeline to identify the lane boundaries in a video. 
The deliverables could be accessed here:
- [README](https://github.com/timotdsantos/CarND-Advanced-Lane-Lines/blob/master/README.md) - this is the writeup and rubric discussion
- [Advanced_Lane_Lines.ipynb](https://github.com/timotdsantos/CarND-Advanced-Lane-Lines/blob/master/Advanced_Lane_Lines.ipynb) - contains the codes 
- [video_output.mp4](./video_output.mp4) - the final video output with lane boundaries

---

### Advanced Lane Finding Project

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
[image_birds_eye]: ./output_images/birds_eye_view.png "Perspective Transform"
[image_binary_birds_eye]: ./output_images/binary_birds_eye.png "Perspective Transform"
[image_sliding_window]: ./output_images/sliding_window_line_detection.png "Sliding Window Search"
[image_roi]: ./output_images/image_roi.png "ROI Search"
[image_lane]: ./output_images/lane_aug.png "ROI Search"
[image5]: ./examples/lane_aug.jpg "Fitted line on Image"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup 

#### 1. This [README](https://github.com/timotdsantos/CarND-Advanced-Lane-Lines/blob/master/README.md) is a detailed writeup of the project.  The project notebook which contains the code may be found here [Advanced_Lane_Lines.ipynb](https://github.com/timotdsantos/CarND-Advanced-Lane-Lines/blob/master/Advanced_Lane_Lines.ipynb) and will be referenced in the discussion. 

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

The code implementation is available in code cell [17].
Here is the output of the pipeline where each color is the contribution of the various color transform layers.
![alt text][image_pipeline]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in cell [20].  The `warper()` function takes as input the undistorted image (`img`) from the `undistort()` function, as well as source (`src`) and destination (`dst`) points.  I chose a hardcoded source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 545,485       | 450,0         | 
| 735,485       | 750,0         |
| 1065,695      | 750,720       |
| 245,695       | 450,720       |

```python
def warper(img, 
           src=np.float32([(545,485),
                  (735,485), 
                  (1065,695),
                  (245,695) 
                  ]), 
           dst = np.float32([(450,0),
                  (1200-450,0),
                  (1200-450,720),
                  (450,720)
                  ])
          ):

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped,M,Minv
```

It uses `cv2.getPerspectiveTransform()` to get the following transform matrices, which is basically used to project the image from one perspective to another:
- **transform matrix (M)** is used to transform images from normal view to Birds-Eye-View
- **inverse transform matrix(Minv)** is used to transform images from Birds-Eye-View back to normal view, this is used for generating the lane overlay and bring it back to augment the original image

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here, I used the test image included as sample in the repository (test_images/straight_lines1.jpg).

![alt text][image_birds_eye]

Here is an example of the binary image that's transformed to its birds-eye-view version.

![alt text][image_binary_birds_eye]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two methods I implemented for of the lane-line detection:
- Sliding Window Search
- Region of Interest Update Method


#### The Sliding Window Search
The sliding window method is best implemented when no line has been detected yet, or the previous detections have been rejected. We run the search on the Bird's Eye View of the Binary Image.

**These are the main parts which can be found in code cell [25] and the reusable function sliding_window() can be found in [26]:**

**i. Find the peaks in the pixel histogram for the left and right halves of the image** - ideally there are 2 peaks for each line which can be found in the left and right half of the image
```python
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```
**ii. Use a sliding window as mask, create a list of pixels belonging to left and right line.** -  Starting from the histogram peak. From the mask, collect the nonzero pixels that belong to the left and right lines 


**iii. Fit a second order polynomial to each line**

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

Below are the histogram and illustration of the sliding window search.
![alt text][image_sliding_window]

#### Region of Interest Update Method
In this method, we use the previously identified points (and fitted as lines) as starting point for scanning nonzero pixels. 
We run the search on the Bird's Eye View of the Binary Image. 

These are the main parts which can be found in code cell [27]. The reusable function update_lanes() can be found in [28]:

**i. Look for nonzero pixels from each line pixels from the previous frame** 
```python
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
left_fit[1]*nonzeroy + left_fit[2] + margin))) 

... do for right line
```

**ii. Fit a second order polynomial to each line**

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

The ROI method is faster since there are starting points from where the new lines could be searched. 
Below is the line detection using ROI.

![alt text][image_roi]


#### Line Class

The class `Line()` was created to aid the detection of lines and for storing various line characteristics, it can be seen in the code cell [32]:
```python
class Line():
    def __init__(self):
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #x values for fitted line
        self.fitx = None
        #y values for fitted line
        self.ploty = None
```

#### LANE Class

The class `Lane()` was created to store the left and right lines, store other characteristics that will be necessary when processing succeeding frames in the video, and integrating the steps in the lane detection workflow. It can be seen in the code cell [33].

```python
class Lane():
    def __init__(self):
        # left line and its attributes
        self.left_line = Line()
        # right line and its attributes
        self.right_line = Line()
        # the final image to be displayed/returned, contains lane visualization and attribute details (curvature, offset)
        self.out_image = None
        # transformation matrix, will be used for birds-eye-view transformation
        self.mtx = None
        # inverse transformation matrix, will be used for transforming lanes from birds-eye-view to normal view
        self.inverse_M = None
        # the current unprocessed image
        self.image=None
        # trigger for sliding window method
        self.detected=False
        # consecutive times the previous fit was reused or the lines failed sanity checks
        self.n_fail=0
        # the mean distance of each of the points in left and right lines
```

**Here are the other methods in the `Lane()` class.** It updates the lane attributes and output images, codes can be found in the code cell [33].

```python
#get left and right line curvature and update the radius_of_curvature attribute of the left and right lines
get_curvature(self)

#augment the original image with the lane overlay and update the out_image of the lane object        
set_lanelines(self,tint=(0,255,0))
    
#get the left and right lines and update the relevant attributes attributes using the ROI method               
update(self,image)
    
#Refresh the lane overlay based on current line attributes - for debugging purpose, use Red
bootstrap_lane_lines(self,image)

#Refresh the lane overlay based on current line attributes    
refresh_lines(self,image,tint=(0,255,0))

#get the left and right lines and update the relevant attributes attributes using the sliding window search                  
sliding_window(self,image)


```


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is computed in the `get_curvature()` method of the Lane class in cells [32] and [33]. The pixel values of the lane are scaled into meters using the scaling factors defined as follows:

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meteres per pixel in x dimension

These values are then used to compute the polynomial coefficients in meters and then the formula given in the lectures and in the linked [tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) is used to compute the radius of curvature.

```python
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
``` 

The position of the vehicle is computed by the function `extract_offset()` in cell [33]. The camera is assumed to be centered in the vehicle and checks how far the midpoint of the two lanes is from the center of the image. The x-intercept or the point at the bottom edge of the left and right lines are used to calculate the lane center, while the center of the image is considered as the car center assuming that the camera is mounted at the center of the car.

```python
lane_center = (left_fitx[-1] + right_fitx[-1])/2
car_center = image.shape[1]/2
center_offset = abs(car_center-lane_center) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines cell [34] in the function `return_line()`.  Here is an example of my result on a test image:

![alt text][image_lane]


---

### Pipeline 

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video_output.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Among the challenges that were encountered were the following:
- non-uniform road color
- non-uniform light and shading conditions (e.g. tree/leaves shading)
- line is reflected on adjacent car
- adjacent car is not completely filtered out by the masks
- discontinuous line (dashed-line)
- gutter/road-edge gets detected as a lane
- other noise

The problems above causes false-positives in the detected pixel points. The main approach was to reject these 'faulty' or 'suspect' fitted lines, and revert to a previously accepted lines. This is acceptable because cameras usually take a short time between frames (at 23 fps, that's 1/23 second between frames), and it's improbable to have drastic change in curvature or lane

**SANITY CHECKS** . were done to accept or reject the detected pixels and the fitted line that is produced. It can be found in code cell [33] and [35] in the `update_()` and `process_image` methods. 

#### Here are rules to be checked:
- Mean distance between left and right fitted lines should be within acceptable range based on the expected lane widths
- The width of the bottom points of the line (closest to the car) are within acceptable range based on expected lane width
- Radius of curvature of left and right lines have a minimum threshold, based on expected lane curvature
- The standard deviation of distance between left and right points are below a threshold. Higher standard deviation means there might be a lot of noise other than the line
- The radius of curvature for the right and left lines should be close to each other since the lines are supposed to be parallel.
- Lane width of top edge should be close to the width of the bottom edge

**Mitigation**
- When the fitted line do not pass the sanity check, the current detected line is discarded and the previous fitted lines are used
- The number of consecutive discarded fitted lines are logged. If it reaches a certain number (i.e. 20 consecutive undetected lines, in a video with 23 fps this is less than 1 second worth), the **sliding window search method** is triggered instead of the ROI method.
- The ROI method uses the variable **margin** to check N number of pixels on the right and left of the previous fitted line. In instances where there's an adjacent car, there's chance that the adjacent car may be detected. In order to fix this, the margin was chosen to be at an optimal width to minimize false positives.
- The coordinates of the polynomial fit is not updated using the current detected line. To prevent erratic change in the polynomial coordinates, an exponential weighted average function is applied. This practically gets the average of the past instances, but gives more weight to the current coordinates. This approach was chosen give more importance to the most recent detection, and to avoid having to keep track of a past coefficients. The formula for the EWMA is shown below:

``EWMA(t)=λY(t)+(1−λ)EWMA(t−1) ; λ=0.7 ; for t=1,2,…,n ``


Other possible improvements:
- we could use a feedback loop or look into control theory to prevent erratic lane fitting output
- other color channels and preprocessing may be considered to be robust to noise and various lighting condition
- the transformation points (src, dst) could be made adaptive using lane/edge detection instead of hard coded value

