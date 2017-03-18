# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* model_brightness.h5 containing a trained convolution with brightness augmentation intended to generalize for the mountain track
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

My final model doesn't include dropout ayers since I realized that the model performs better when it doesn't include dropout.

I've tried this approach but unfourtunately it perform way worst that training on less amount of epochs without dropout.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 89).

#### 4. Appropriate training data

I used the sample data provided by Udacity and it happen to work just well.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow Nvidia architecture, I've tried alterating that model but I ended up getting worst performace that the original one.

I stayed with that same model since it worked just well with the sample data and the car was able to make a complete lap flawlessly.

I've tried to add more epochs, bu the model started to overfit, so I left the amount of epochs in three. I also tried to add dropout with more epochs, but the model had a worst performance.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-87) consisted of a convolution neural network with the following layers and layer sizes:

1. First a cropping layer
2. A normalization layer
3. Convolution 25x5x5, subsampling 2x2 relu activation
4. Convolution 36x5x5, subsampling 2x2 relu activation
5. Convolution 48x5x5, subsampling 2x2 relu activation
6. Convolution 64x3x3, subsampling 2x2 relu activation
7. Convolution 64x3x3, subsampling 2x2 relu activation
8. Flattening
9. Dense 100
10. Dense 5
11. Dense 1

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![model architecture][model.png]

#### 3. Creation of the Training Set & Training Process

Since the saple data provided by Udacity performed well, I didn't need to make data.

For data augmentation:

1. I shuffled the images
2. I used left camera imagges applying a correction of 0.2
3. I used right camera imagges applying a correction of -0.2 
4. I flipped the images and reversed the steering angle for those
5. Finally I randomly changed the brightness of the images

I used a generator so that I didn't had to hold all of the data in memory.
This made the trining process somehow faster.

I used this training data for training the model.
The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 3 as evidenced, after that, the model started to overfit.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

Even though I tried using dropout and enlarging the amount of epochs, my model wasn't able to generalize in the mountain track.
The model without brightness augmentation happen to drive better that the one without brigthness augmentation (model.h5 vs model_brightness.h5)
