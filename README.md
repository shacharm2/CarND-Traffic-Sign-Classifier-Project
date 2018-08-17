## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, a structured convolutional network followed by a fully-connected network head learns the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The model is trained, validated and tested on the dataset and finally, evaluated on a small set of web images.

To meet the specifications, the project includes three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown and a pdf file 

[1_classes_distribution]:output_images/1_classes_distribution.png
[1_raw_images]:output_images/1_raw_images.png
[1_raw_samples]:output_images/1_raw_samples.png
[1_test_ordered_hist]:output_images/1_test_ordered_hist.png
[1_train_ordered_hist]:output_images/1_train_ordered_hist.png
[1_valid_ordered_hist]:output_images/1_valid_ordered_hist.png
[2_postprocessing_2_samples]:output_images/2_postprocessing_2_samples.png
[2_rgb_and_postprocessed]:output_images/2_rgb_and_postprocessed.png

[2_postprocessing]:output_images/2_postprocessing.png
[2_P5max_testset]:output_images/2_P5max_testset.png
[3_P5max_webimages_probabilities]:output_images/3_P5max_webimages_probabilities.html
[3_P5max_webset]:output_images/3_P5max_webset.png



# The Project
---

## Intro

The project consists of a deep neural network consisting of a few convolutional layers, stacked as both sequential and parallel inception layers to which attaches a fully connected classification network. 


## Goals

The steps of this project are presented as the following:

* Load the data set
* Dataset expolration and summerization
* Designing training and testing of the model architecture
* Using the model to make predictions on new images from the web
* Analyzing the softmax probabilities of the new images
* Summarizing the results with a written report




# Dataset

## Dataset classes
The [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) has 43 classes, for which a few examples of each have been plotted:


## Dataset samples
![1_raw_samples][1_raw_samples]

And in order to see some of the variance, a few selected classes have been plotted, in order to see variations is size, luminosity, backgrounds and interference of other classes, such in the case of a part of another traffic sign within the image
![1_raw_images][1_raw_images]

## Dataset disribution

Important criteria
* number of examples
* balance

The disribution of the classes between the train, test and validation can be show here. It can be seen the data distribution is unbalanced 

![1_classes_distribution][1_classes_distribution]

Looking at the individual classes distirbution shows the spectrum of frequency of each class

![1_train_ordered_hist][1_train_ordered_hist]

in parallel with the validation & test distributions, where similar balance exists for the various classes. 
![1_test_ordered_hist][1_test_ordered_hist]

However, this unbalanced , in turn, might cause a bias in the results, where most train is done, for example on the 50mph limit traffic sign and tested mostly on it and thus, not masking out lack of data on less frequent signs, while presenting accuracy that maight not scale well to the real world.

# Image preprocessing

Image preprocessing was done using 
    
    PreprocessingPipeline class()
        ...
        @staticmethod
        def whiteBalance(image):
            ...
        @staticmethod
        def normalizeRGB(image):

        @staticmethod
        def equalizeRGBHist(image):
            ...

        def run(self, image, image_class=None):
            # postprocessing and augmentation
       

The pipeline:
1. White balance in LAB colorspace [1]
2. Histogram YUV colorspace equalization [2]
3. Normalize RGB to the [0.1, 0.9] independently
4. Augmentation

    4.1. Flip left/right to the matching classes (stop sign)
    
    
 
![2_postprocessing_2_samples][2_postprocessing_2_samples]

![2_rgb_and_postprocessed][2_rgb_and_postprocessed]

![2_postprocessing][2_postprocessing]

# References
2_postprocessing_2_samples.png
[1] [GIMP's White balance](https://pippin.gimp.org/image-processing/chapter-automaticadjustments.html)

[2] [YUV histogram equalization](https://chrisalbon.com/machine_learning/preprocessing_images/enhance_contrast_of_color_image/)