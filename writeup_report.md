# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/run_result.png "Training Summary"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model does use convolutional neural network with 5x5 / 3x3 filter sizes and depths between 24 and 64 (model.py lines 85-89).  The model includes RELU activation function to introduce nonlinearity (passed as a parameter to convolutional and fully connected layers), and the data is normalized in the model using a Keras lambda layer (code line 82). 

#### 2. Attempts to reduce overfitting in the model

The model utilizes regularization in order to reduce overfitting (model.py line 85). 

The model was trained and validated on different data sets (80/20 split) to ensure that the model was not overfitting (code line 22).  The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a slow center lane driving and used inputs from all three cameras (center, left, right).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to slowly increase the complexity of the model while observing the behavior of the trained model in the simulator.

My first step was to use a convolutional neural network model similar to the NVIDIA model described in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).  I thought this model might be appropriate because it was specifically designed for the problem I am trying to solve.

For the first run I did not use the left and right cameras.  However, I did augment the training samples with a mirror image of the center image.  I negated the steering input in order to match the mirror image samples.  The image and steering angle data was split into a training and validation set in a 80/20 split.  The model had low Mean Square Error(MSE) on both the training and validation set.

When the model was run through the simulator, it quickly became clear that the model will not be able to complete the track.  The car was driving okay on the straight-aways, but quickly got stuck in the corner and was not able to recover.

The MSE was low but the model did not perform well, suggesting that perhape there weren't enough samples for training.  At this point I decided to collect more samples.  I started with one very slow lap on the track, which resulted in 14043 samples.  While this model seemed to perform better, it still went off the road and got stuck eventually.  The problem seemed to be in the inability to turn correctly in the corners.

In order to teach the car to turn better, I decided to add in the left and right cameras.  Based on visual examination of the steering input and how the left and right camera images looked, I estimated that the steering correction was about 0.25.  So the training set was augmented with the left camera image and +0.25 steering as well as the right camera image and -0.25 steering compared to the center image steering.

This time the car made its way around the track, but did cross the yellow line a few times and came close to going off the road.  Encouraged by the improvement, I increased the steering correction to 0.5, 0.75, 1, then eventually to 2.  After this correction, the vehicle is able to drive autonomously around the track without leaving the road.

Even though my model did not seem to be overfitting, I added regularization to the first convolutional layer because it was required in the project rubric.  I also increased the steering correction to 5 as the model with regularization seemed to have more difficulty navigating the track.  The final model successfully drives through the track.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-102) consisted of a convolution neural network with the following layers and layer sizes:
| Layer Type | Activation  | Size/Description |
|----------|:-------------:|------:|
| Convolutional | ReLU | 5x5 filter, 2x2 subsample, 24 depth, regularization on the weights |
| Convolutional | ReLU | 5x5 filter, 2x2 subsample, 36 depth |
| Convolutional | ReLU | 5x5 filter, 2x2 subsample, 48 depth |
| Convolutional | ReLU | 3x3 filter, 1x1 subsample, 64 depth |
| Convolutional | ReLU | 3x3 filter, 1x1 subsample, 64 depth |
| Flatten		| None | |
| Fully connected/dense | ReLu | 100 |
| Fully connected/dense | ReLu | 50 |
| Fully connected/dense | ReLu | 10 |
| Output layer | None | 1 |


#### 3. Creation of the Training Set & Training Process

The plan was to gradually capture more training data after observing the model behavior.  I started out with one very slow lap around the track, intending to capture several more as needed.  I had also planned to record the vehicle recovering from the left and right side of the road.

Howver, the car was able to successfully complete the track using only the single slow lap.  Further training data was not collected.

The slow lap consisted of 14,043 data points. I then preprocessed this data by cropping the top 50 pixels (sky) and bottom 10 pixels (car hood).  Further, the image was normalized: divide by 255 and subtract 0.5.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was perhaps 3 as evidenced by the image below.  While the model improved significantly between epoch 1 and 2, it only marginally improved between epoch 3 and 4.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

![image1]