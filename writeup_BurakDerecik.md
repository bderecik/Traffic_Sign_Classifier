# **Traffic Sign Recognition** 

## Desciption: This projects aims to create classifier to detect traffic signs.

### Introduction.

In this project, I develop a deep neural network (DNN) model (convolutional neural networks) to classify traffic signs.  [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is used to train and validate the proposed DNN model in this project. 

To show capability of the developed model, received traffic signs from the internet are tested with fully trained DNN model.

---

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/test.png"  height ="40%" width="60%">

**The steps of this project are the following:**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

***
#### 1. Raw Data Loading

Raw data has been imported via pickle.

Some samples of raw data can be seen as following:
***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/readme_imgs/rawdata.png"  height ="40%" width="60%">

***

### Data Set Summary Exploration
***
#### 2. Summary of Data
* The size of training set is 139196
* The size of the validation set is 17640
* The size of test set is 50520 
* The shape of a traffic sign image is 32x32x1
* The number of unique classes/labels in the data set is 43

Distribution of image classes have been plotted as following:

#### Train
***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/train.png"  height ="40%" width="60%">

***
#### Test
***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/test.png"  height ="40%" width="60%">

***
#### Validation
***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/validation.png"  height ="40%" width="60%">

***
#### 3. Pre-Processing the Image

I have implemented following pre-process techniques:

* Normalize and convert to grayscale all dataset (train,valid,test)
* Flipping
* Sharping
* Random rotating 

Then, I applied concatenating to merge all pre-processed images. Also,datasets are shuffled in order to elimate the influence of the order of the data.

Finally, some samples of pre-processed images can be seen as following:

***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/readme_imgs/preprocessed_image.png"  height ="40%" width="60%">

***

#### 4. Model

I have modified the original LeNet architecture. I have added 1 conv layer(layer3) 3x3 filter and 1x1 stride and 1 fully conv layer to the original LeNet model.

##### Parameters:

* The learning rate: 0.001
* The optimizer: AdamOptimizer
* The EPOCHS: 100
* The BATCH SIZE: 128

##### Accuracy:

* Train Accuracy = 0.994
* Valid Accuracy = 0.953
* Test Accuracy = 0.929

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 3x3x32					|
| RELU					|												|
| Fully connected		| Input:288 Output:100        									|
| RELU					|												|
| Dropout					|			keep prob:1									|
| Fully connected		| Input:180 Output:120        									|
| RELU					|												|
| Dropout					|			keep prob:1									|
| Fully connected		| Input:120 Output:84        									|
| RELU					|												|
| Dropout					|			keep prob:1									|
| Fully connected		| Input:84 Output:43        									|

Based on given parameters and the architecture, the validation accuracy converged around 95%, the validation vs train accuracy can be seen as following:

***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/readme_imgs/train_acc.png">

***

In iterative process, I firstly used LeNet, however, due to absance of dropout layers, model is getting overfitting.

I added more conv and fully connected layer and dropout layers. So, I have seen that number of learning parameters are increased.

Also, I have tuned dropout, number of images in dataset, learning rate, batch size and epoch. I obtained best validation accuracy using specified parameters at above.

I can say that adding dropout helps me to handle overfitting.

My model is suitable for traffic sign application since it does not affect computational cost.

Final model's accuracy on the training, validation and test set provide evidence that the model is working well since I obtained good enough accuracy values.

#### 5. Test

I have found 5 images from the web,which can be seen at below.And the testing accuracy is 60% for those images (shown below):

***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/readme_imgs/test_images.png">

***

##### Output of performance analysis:
```
INFO:tensorflow:Restoring parameters from ./lenet
Image 1
Image Accuracy = 0.000

INFO:tensorflow:Restoring parameters from ./lenet
Image 2
Image Accuracy = 0.500

INFO:tensorflow:Restoring parameters from ./lenet
Image 3
Image Accuracy = 0.667

INFO:tensorflow:Restoring parameters from ./lenet
Image 4
Image Accuracy = 0.750

INFO:tensorflow:Restoring parameters from ./lenet
Image 5
Image Accuracy = 0.600

```

##### Softmax probabilities for each prediction.


***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/readme_imgs/softmax.png">

***

Finally, the softmax probabilities have been calculated for the images and can be seen at below:

***

<img src="/home/workspace/CarND-Traffic-Sign-Classifier-Project/readme_imgs/softmax_prob_bar.png">

***