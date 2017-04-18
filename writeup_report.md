# **Behavioral Cloning**

## **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[img_model]: ./images/p3-model.png "Model Visualization"
[img_loss]: ./images/p3-lossgraph.png "MSE Loss Graph"
[img_centered]: ./images/p3-centered.jpg "Driving Centered Example"
[img_far_out]: ./images/p3-far_out.jpg "Driving Far Off Center"
[img_correcting]: ./images/p3-correcting.jpg "Driving Correcting to Center"
[img_final_correction]: ./images/p3-corrected.jpg "Driving Final Correction"
[img_unflipped]: ./images/p3-centered.jpg "Driving Unflipped Example"
[img_flipped]: ./images/p3-flipped.jpg "Driving Flipped Example"
[img_cropped]: ./images/p3-cropped.jpg "Cropped Image Example"

[//]: # (Link references)

[model_python]: https://github.com/Jazzbert/CarND-Behavioral-Cloning-P3/blob/master/model.py "Github model.py"
[drive_python]: https://github.com/Jazzbert/CarND-Behavioral-Cloning-P3/blob/master/drive.py "Github drive.py"
[model_h5]: https://github.com/Jazzbert/CarND-Behavioral-Cloning-P3/blob/master/model.h5 "Github model.h5"
[writeup_file]: https://github.com/Jazzbert/CarND-Behavioral-Cloning-P3/blob/master/writeup_report.md "This write-up!"
[final_video]: https://github.com/Jazzbert/CarND-Behavioral-Cloning-P3/blob/master/video.mp4 "Github video"
[nvidia_blog]: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ "Nvidia self-driving model"
[first_model]: https://github.com/Jazzbert/CarND-Behavioral-Cloning-P3/blob/master/first.h5 "Kept my first try...just for laughs!"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py][model_python] containing the script to create and train the model
* [drive.py][drive_python] for driving the car in autonomous mode
* [model.h5][model_h5] containing a trained convolution neural network
* [writeup_report.md][writeup_file] summarizing the results
* [video.mp4][final_video] showing my _victory_ lap :)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

Additionally, a new model can be trained using the model.py.  While the process for using the training data is listed below, this model should work for any data trained using the driving simulator and placed in a folder './data/' relative to the working directory.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [NVIDIA convolution neural network][nvidia_blog] with the addition of two dropout layers after the first two fully-connected layers.  The model also starts with a pre-processing layer to normalize and center the values around zero.

Following the original NVIDIA model, each convolution includes a RELU activation layer.  This model performed great right at the start.  Adding dropout layers improved performance substantially, allowing the autonomous mode to make it all the way around the track--albeit slowly and a bit roughly.

#### 2. Attempts to reduce overfitting in the model

In addition to dropout mentioned above.  The loss functions on training and validation were noted as per the image below to limit the number of epochs to seven before overfitting would become a problem.

![alt text][img_loss]

I also included data from the second training track to help the model generalize further. The data was shuffled for each batch to make sure there were no sequential connections between each sample image.

A standard, random 20% validation set was taken from the original full data set to measure the training results.  Ultimately the truth, is seeing the car being able to drive around the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.  

#### 4. Appropriate training data

Simply put, the data provided to the model showed the behavior of getting to the center of the road.  While my children volunteered to drive the simulator, they would have as much fun crashing and driving into the lake as keeping to the center lane.  

Restated, goal was to only include data that was showing the car staying in the center of the lane or the car moving towards the center from a slightly off or more unusual position.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Frankly, we all stand on the shoulders of giants.  After getting a very basic model connected to make sure everything was working, going to the NVIDIA _self-driving car_ convolution model seemed to be a no-brainer.  And it worked great!  With just a few minor tweaks, it was very easy to get the simulator to drive around the track successfully.  

Different aspects of building the model are noted above with respect to avoiding overfitting and tweaking the model in general.  Detail around how I provided training data is listed above and I'll go into further detail in subsection 3 below.

There is, of course room for improvement given additional time spent.  Some examples of where I would further improve include:
* Experimenting with tweaking the learning model
* Try a different base model altogether
* Add/change approaches to training data (maybe letting my kids run amok could produce better results!)
* Large and unstable oscillations start to occur at speeds above 25.  I believe some of the drive.py could be tweaked to use the fitted results better including smoothing changes.
* Some oscillation behavior may also be addressed by providing better (more gradual) examples of recovery from an unusual attitude.
* Get more training data from track 2, especially in troublesome areas (e.g., shadowed areas and hard curves)

#### 2. Final Model Architecture

Boom...see below.  The code clearly lists out the model in model.py including the normalization layer (line 64), convolution layers (lines 67-72), and the fully connected layers (lines 73-78).  Frankly, after learning TensorFlow, I _love_ Keras for it's readability and simplicity in execution.

![alt text][img_model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving as well as another lap going the oposition direction.  Here is an example image of center lane driving:

![alt text][img_centered]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to give examples when the model gets more than just a little off to the side. I showed these images from multiple places around the track to include different side types (including the bridge) and different road curve angles.  Here's a sequence showing starting attitude, corrective adjustment, and readjustment when reaching center:

![alt text][img_far_out]
![alt text][img_correcting]
![alt text][img_final_correction]

Then I repeated this most of this process on track two in order to get more data points.  In addition, on track 2 I gathered some data for the shadowed area in attempt to get better autonomous success on track 2.

To augment the data sat, I also flipped images and angles thinking that this would both increase the amount of sample data and allow the model to generalize further. For example, here is an image that has then been flipped:

![alt text][img_unflipped]
![alt text][img_flipped]

After the collection process, I had 9,897 data points, each with 3 images (center, left, right), all of which were doubled.  The left and right images were used with a slight steering  correction angle of +/- 0.2 degrees.  With 20% of the total images taken for validation, this resulted in 59,382 samples for training.

Prior to submitting to model (which included it's own normalization/centering pre-processing), I trimmed the top and bottom from the images to reduce the amount of "noise" that would not likely have improved results:

![alt text][img_cropped]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was seven as evidenced by the increase in validation loss with a greater number epochs as noted above.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Conclusion

I really enjoyed this project.  It was truly amazing even the [first time I connected a trained model][first_model] and the car just drove in a rough/crazy circle.  Thanks for taking a look at my project!!
