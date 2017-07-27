# Udacity-Project-1 O-Ali

# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Use the pipeline to identify lanes in videos
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./Images/CannyEdge/CannyEdges_solidWhiteCurve.jpg "solidWhiteCurve.jpg"
[image2]: ./Images/CannyEdge/CannyEdges_solidYellowCurve.jpg "solidYellowCurve.jpg"

[image3]: ./Images/Grayscale/GrayScale_solidYellowCurve.jpg "solidYellowCurve.jpg"
[image4]: ./Images/Grayscale/GrayScale_solidWhiteCurve.jpg "solidWhiteCurve.jpg"

[image5]: ./Images/HSLMask/mask_white_solidWhiteRight.jpg "solidWhiteRight.jpg"
[image6]: ./Images/HSLMask/mask_white_solidWhiteCurve.jpg "solidWhiteCurve.jpg"

[image7]: ./Images/HSLMask/mask_yellow_solidYellowCurve.jpg "solidYellowCurve.jpg"
[image8]: ./Images/HSLMask/mask_yellow_solidYellowCurve2.jpg "solidYellowCurve2.jpg"

[image16]: ./Images/FinalResult/solidYellowCurve.jpg "solidYellowCurve.jpg"
[image9]: ./Images/FinalResult/solidWhiteCurve.jpg "solidWhiteCurve.jpg"
[image10]: ./Images/FinalResult/solidWhiteRight.jpg "solidWhiteRight.jpg"
[image11]: ./Images/FinalResult/solidYellowCurve2.jpg "solidYellowCurve2.jpg"
[image12]: ./Images/FinalResult/solidYellowLeft.jpg "solidYellowLeft.jpg"
[image13]: ./Images/FinalResult/whiteCarLaneSwitch.jpg "whiteCarLaneSwitch.jpg"

[image14]: ./Images/grayANDhsl/GandHSL_solidYellowLeft.jpg "solidYellowLeft"
[image15]: ./Images/grayANDhsl/GandHSL_solidWhiteCurve.jpg "solidWhiteCurve.jpg"

---

### Reflection

### 1. The Pipeline

**Step 1 - Grayscale

My pipeline consists of 6 steps, with the first being the conversion to grayscale.

The grayscale image will be combined with the masks and used for the canny edge detection later in the process.

#cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

![alt text][image3] ![alt text][image4]

**Step 2 - HSL Mask

Next is creating the mask to identify the the lanes by color more easily.
I convert the image to HSL and set the boundaries to isolate white and yellow. HSL allows for easy identification of the colors even in more difficult shaded situations.
	#White Mask Boundaries:
	lower_white = np.array([0,200,0], dtype = "uint8")
	upper_white = np.array([255,255,255], dtype = "uint8")

	#Yellow Mask Boundaries:
	lower_yellow = np.array([10,0,100], dtype = "uint8")
    	upper_yellow = np.array([40,200,255], dtype = "uint8")

## White Mask
![alt text][image5] ![alt text][image6]
## Yellow Mask
![alt text][image7] ![alt text][image8]

After obtaining the seperate masks I perform a bitwise OR to combine them into one image
	#mask = cv2.bitwise_or(mask_white,mask_yellow)
Then a bitwise AND with the grayscale image
	#output = cv2.bitwise_and(gray,gray,mask=mask)

![alt text][image14] ![alt text][image15]

**Step 3 - Canny Edge Detection

The Canny edge detection is done by first passing the output of the previous step through a gaussian blur with value 13
I set the low threshold to 50 and the high to 150 and call the cv2.Canny on the blurred image with those values.

	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	
#Here are a couple examples for the result:

![alt text][image1] ![alt text][image2]

**Step 4 - region of interest

Next in the pipeline is creating the region of interest and masking out the rest of the image.
rows and cols are set to the shape sizes of the 'edges' image (result of the Canny function). The rest of the variables are set by being a percent of the
total so the region of interest scales with the image size.

	rows, cols = edges.shape[:2]
	bottomLeft = [cols*.1, rows*.97]
	topLeft = [cols*.4, rows*.6]
	topRight = [cols*.6, rows*.6]
	bottomRight = [cols*.9, rows*.97]

	vertices = np.array([[bottomLeft,topLeft,topRight,bottomRight]], dtype=np.int32)

	masked_edges = region_of_interest(edges, vertices)


**Step 5 - Hough
After masking the image with the region of interest it's time to run the Hough transform and get the hough lines. I set the variables and call the 
cv2.HoughLines function with the 'masked_edges' image

	rho = 1 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 20     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 20 #minimum number of pixels making up a line
	max_line_gap = 300   # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on

	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),minLineLength=min_line_len,maxLineGap=max_line_gap)

**Step 6 - Extrapolate and Draw Lines


The functions below are used to consolidate and extrapolate the detected hough lines and they are called through the pipeline with

	lines_edges = draw_lane_lines(image, lane_lines(image, lines))
	
	"""
	average_slope_intercept calculates the average slope of the lines on the left and the lines on the right to create 
	the best slope for our final line
	"""

	def average_slope_intercept(lines):
	    left_lines    = [] # (slope, intercept)
	    left_weights  = [] # (length,)
	    right_lines   = [] # (slope, intercept)
	    right_weights = [] # (length,)

	    for line in lines:
		for x1, y1, x2, y2 in line:
		    if x2==x1:
			continue # ignore a vertical line
		    slope = (y2-y1)/(x2-x1)
		    intercept = y1 - slope*x1
		    length = np.sqrt((y2-y1)**2+(x2-x1)**2)
		    if slope < 0: # y is reversed in image
			left_lines.append((slope, intercept))
			left_weights.append((length))
		    else:
			right_lines.append((slope, intercept))
			right_weights.append((length))

	    # add more weight to longer lines    
	    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
	    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None

	    return left_lane, right_lane # (slope, intercept), (slope, intercept)

	"""
	make_lane_points returns the points to create the line(pixels) using the input from lane_lines
	"""

	def make_line_points(y1, y2, line):
	    if line is None:
		return None

	    slope, intercept = line

	    # make sure everything is integer as cv2.line requires it
	    x1 = int((y1 - intercept)/slope)
	    x2 = int((y2 - intercept)/slope)
	    y1 = int(y1)
	    y2 = int(y2)

	    return ((x1, y1), (x2, y2))

	"""    
	lane_lines(also in the main pipeline) takes the original image and the image with the hough lines draw
	it calls the average_slope_intercept to get the average slopes, sets two variables(points) to the bottom of the 
	image and slightly below the middle, and calls the make_line_points function for both left and right lanes
	"""

	def lane_lines(image, lines):
	    left_lane, right_lane = average_slope_intercept(lines)

	    y1 = image.shape[0] # bottom of the image
	    y2 = y1*0.6         # slightly lower than the middle

	    left_line  = make_line_points(y1, y2, left_lane)
	    right_line = make_line_points(y1, y2, right_lane)

	    return left_line, right_line

	"""
	The function draw_lane_lines is called in the main pipeline with the original image and the result of the
	lane_lines function, it returns sum of the image's weights
	"""    
	def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
	    # make a separate image to draw lines and combine with the orignal later
	    line_image = np.zeros_like(image)
	    for line in lines:
		if line is not None:
		    cv2.line(line_image, *line,  color, thickness)
	    # image1 * a + image2 * ß + ?
	    # image1 and image2 must be the same shape.
	    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

The results are these images:

![alt text][image16] ![alt text][image9]
![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13]

The pipeline can also be run on videos which can be viewed in the 'VideosOutput' directory.

### 2. Potential shortcomings with your current pipeline


A Potential issue with this pipeline is when a line with the color white or yellow is on the road and is almost vertical, the average slope
will be calculated incorrectly and cause the lines to misbehave for a second.

Another shortcoming, related to the above, is in regards to strong angles on the lane lines, again it will cause the average slope to present the line
incorrectly.


### 3. Possible improvements to your pipeline

A possible improvement would be to implement a way to draw the lines shorter so the angle or curve of the lane line doesnt affect the entire line.

Another potential improvement could be to find the optimal values and color scheme to use when creating the masks in step 2
