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
[2_rgb_and_postprocessing]:output_images/2_rgb_and_postprocessing.png
[2_postprocessing]:output_images/2_postprocessing.png
[2_P5max_testset]:output_images/2_P5max_testset.png
[3_P5max_webset]:output_images/3_P5max_webset.png
[udacity_inception]:output_images/udacity_inception.jpg
[3_training_accuracy]:output_images/3_training_accuracy.png
[3_web_images_results]:output_images/3_web_images_results.png


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

## Preprocessing pipeline

Image preprocessing was done using the following class
    
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
1. White balance in LAB colorspace [1]. White balance has been chosen in order to remove unrealistic color casts, so that objects which appear white in person are rendered white in the image [2]
2. YUV-colorspace histogram equalization [3] for contrast improvment
3. Normalize RGB to the [0.1, 0.9] independently, as suggested in Udacity slides, to avoid zero values.
4. Augmentation

    4.1. Flip left/right to the matching classes (e.g. stop sign)

    4.2. Flip up/down for matching classes (e.g. 30 mph limit, or stop sign)

    4.3. Flip up/down and left right for matching classes (e.g. Priority road, No vehicles, End of all speed and passing & roundabout mandatory)

    4.4. flip left/right and change class (e.g. Dangerous curve left and Dangerous curve right)

    4.5. Shifts - small random shifts in x,y directions (image coordinates)

    4.6. Rotation - point-of-view rotation about y axis (image coordinates), such that changes the view of the observer. Most signs would be observed from the same hight, but from different angle. Thus, instead of random projections, a rotation about the y axis has been selected to represent a physical real world augmentation case.

5. RGB output - RGB was chosen instead of grayscale, since grayscale would lose intricate information that exists between the three original channels.

## Preprocessing pipeline side by side

Two examples of the post processing are presented, either wth low or high brightness and their post processed counterparts:
 
![2_postprocessing_2_samples][2_postprocessing_2_samples]

## Preprocessing pipeline with augmentation
To further illustrate the examples, the preprocessed images are shown with their matching augmentations 

![2_rgb_and_postprocessing][2_rgb_and_postprocessing]

## Additional preprocessing results

![2_postprocessing][2_postprocessing]


# Model

## Model architecture

The model architecture is composed of three main parts; a deep convolution section, catching features at various sizes; an inception section, aimed at extracting localities at various scales around each point; a fully connected layer for classification.

1. Deep convolution layer
    1.1. InputTwo 3x3 convolution layers, with 16 channels

      * Valid padding
      * ReLU activation

    1.2. A single 1x1 convolution with 16 channels
    1.3. Dropout - 50%
    1.4. Max pooling

2. Inception layer

    2.1. Parallel 5x5, 3x3, 1x1 convolutional layers, with 32 channels

      * Same padding - for inception concatenation 
      * ReLU activation

    2.2. Concatenated output, stacking all channels on top of each other

    2.3. Dropout - 50%
    2.4. Max pooling

3. Fully connected output layer - 
    3.1. Contatenation of pre-max-pool of Layer 1 and of layer 2. 
    ![udacity_inception][udacity_inception]
    
      * This was done in order to bring the entire data of each of the layers' scale to the fully connected output. Concatenating prior to max-pooling also slightly changes the nonlinearity process, in order to slighly lower the correlation with the intermediate layers.
      * ReLU activation
      * Dropout - 50%
    

    3.2 Fully connected of two times the size of the output size (2x43)
      * L2 weights regulizer
      * ReLU activation
    
    3.1. Fully connected logits layer the size out output (43 outputs)
      * L2 weights regulizer

## Model training

Model training was executed as follows. The results have converged fast and to a high rate and not a lot of parameter tweaking and searching had to be done.

* Learning rate at 10<sup>-3</sup>
    * [10<sup>-1</sup>, 10<sup>-4</sup>] have been tested
    * Exponential decay has not been tested
* Adam optimizer
    * Adam has been selected, simular to the LeNet udacity architecture, since it is both effective and easy to use. 
* Regularization
    * Dropout was set at 50%, uniform for all layers
    * Weight L2 regulaizer for the fully connected layers
* 10 Epochs over data

    * The epoch number has been chosen such that no additional validation set accuracy improvement has been seen beyond and 

## Model results

The following graph shows the accuracy results for the training phase on both training and validation:

![3_training_accuracy][3_training_accuracy]

Before contuining to the web test set, a visualization of 5 example from the test set and the top 5 probabilities:

![2_P5max_testset][2_P5max_testset]

## Validatoin and Test sets results

* val acc 0.990
* test acc 0.966


# Web traffic signs 

Images have been found on the web. Some images are public stock images with markings on them. Some are at a different angle, some differ in size - a little further or taken at a close distance (a stop sign from below and very large). Another example - a priority work sign with another circular sign attached behind him forms an unseen shape around the triangular traffic sign shape.

Nonetheless, if these can be easily identified by a human eye, it should be expected to have similar identification capabilities from the network:

![3_web_images_results][3_web_images_results]

For which a detailed image and bar graph can be seen here:

![3_P5max_webset][3_P5max_webset]


3_web_images_results

# References
2_postprocessing_2_samples.png
[1] [GIMP's White balance](https://pippin.gimp.org/image-processing/chapter-automaticadjustments.html)

[2] [Cambridge in color](https://www.cambridgeincolour.com/tutorials/white-balance.htm)

[3] [YUV histogram equalization](https://chrisalbon.com/machine_learning/preprocessing_images/enhance_contrast_of_color_image/)

[4] [Udacity Inception Module](https://www.youtube.com/watch?v=VxhSouuSZDY)