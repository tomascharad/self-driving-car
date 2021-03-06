## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/notcar.png
[hog0]: ./test_output/car_hog0.jpg
[hog1]: ./test_output/car_hog1.jpg
[hog2]: ./test_output/car_hog2.jpg
[hog3]: ./test_output/not_car_hog0.jpg
[hog4]: ./test_output/not_car_hog1.jpg
[hog5]: ./test_output/not_car_hog2.jpg
[hog6]: ./test_output/orig_car_hog0.jpg
[hog7]: ./test_output/orig_car_hog1.jpg
[hog8]: ./test_output/orig_car_hog2.jpg
[hog9]: ./test_output/orig_not_car_hog0.jpg
[hog10]: ./test_output/orig_not_car_hog1.jpg
[hog11]: ./test_output/orig_not_car_hog2.jpg
[test0]: ./test_output/test0.png
[test1]: ./test_output/test1.png
[test2]: ./test_output/test2.png
[test3]: ./test_output/test3.png
[test4]: ./test_output/test4.png
[test5]: ./test_output/test5.png
[heat0]: ./test_output/test_heat0.png
[heat1]: ./test_output/test_heat1.png
[heat2]: ./test_output/test_heat2.png
[heat3]: ./test_output/test_heat3.png
[heat4]: ./test_output/test_heat4.png
[heat5]: ./test_output/test_heat5.png
[test_labeled_heat0]: ./test_output/test_labeled_heat0.png
[test_labeled_heat1]: ./test_output/test_labeled_heat1.png
[test_labeled_heat2]: ./test_output/test_labeled_heat2.png
[test_labeled_heat3]: ./test_output/test_labeled_heat3.png
[test_labeled_heat4]: ./test_output/test_labeled_heat4.png
[test_labeled_heat5]: ./test_output/test_labeled_heat5.png



[image5]: ./examples/sliding_window.jpg
[image6]: ./examples/bboxes_and_heat.png
[image7]: ./examples/labels_map.png
[image8]: ./examples/output_bboxes.png
[video9]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the in the function `get_hog_features` of the file hog_classify.py.  

I start in the file train.py by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car class][image1]
![not car class][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of:
`orient = 9`
`pix_per_cell = 8`
`cell_per_block = 2`
`hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"`
`spatial_size = (32, 32)`
`hist_bins = 32`
`hist_range=(0, 256)`

![alt text][hog0]
![alt text][hog1]
![alt text][hog2]
![alt text][hog3]
![alt text][hog4]
![alt text][hog5]
![alt text][hog6]
![alt text][hog7]
![alt text][hog8]
![alt text][hog9]
![alt text][hog10]
![alt text][hog11]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters but, they tend to have an excelent test accuracy, but they don't detect quite well in general, this was the best I could get.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using hog features, color histogram features and bin spatial features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for the sliding windows is in the `hog_subsampling.py` file in the `find_cars` method, starting from line 25. I decided to scale from 1.0 to 2.0 with a 0.5 offset. I decided this after several iterations looking at how sliding windows were applied. The smallest scale, I just scaned from y=400 till y=460 just to focus on small cars.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I ended up searching in one scale YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided an average result.  Here are some example images:

![alt text][test0]
![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
![alt text][test5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_output/video_state_final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also added a deque of length 10, and averaged the heatmaps in order to soften the labels of the detected cars. This way we also improve the removal of false positives.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six labeled frames and their corresponding heatmaps:

![Heat][heat0]: ./test_output/test_heat0.png
![Heat][test_labeled_heat0]: ./test_output/test_labeled_heat0.png
![Heat][heat1]: ./test_output/test_heat1.png
![Heat][test_labeled_heat1]: ./test_output/test_labeled_heat1.png
![Heat][heat2]: ./test_output/test_heat2.png
![Heat][test_labeled_heat2]: ./test_output/test_labeled_heat2.png
![Heat][heat3]: ./test_output/test_heat3.png
![Heat][test_labeled_heat3]: ./test_output/test_labeled_heat3.png
![Heat][heat4]: ./test_output/test_heat4.png
![Heat][test_labeled_heat4]: ./test_output/test_labeled_heat4.png
![Heat][heat5]: ./test_output/test_heat5.png
![Heat][test_labeled_heat5]: ./test_output/test_labeled_heat5.png

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

The most time consuming part what discovering that I wasn't using all of the training data, so that made my algorithm to not perform well and I spent a lot of time figuring out why.

The pipeline is very robust right now, I think though that all of the traiing data is quite tailored for this pipeline, it would be interesting to evaluate the performance with real world images.

I also will love to know how to make the processing faster so I could apply this to live detection algorithm.

