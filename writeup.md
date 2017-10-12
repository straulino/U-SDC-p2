#**Traffic Sign Recognition** 

In this project we build a Convolution Network and we train it to classify German traffic signs. The aim is to obtain at least 93% accuracy on the validation set, test it on a test set and finally test it on new traffic signs to be obtained from the web.

Below we will give o brief overview of the process we followed to do so, and try to explain why we make the choices we made, as well as where could we potentially improve our Neural Network.





<img src="examples/newImage0.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage1.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage2.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage3.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage4.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage5.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage6.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage7.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage8.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage9.jpg" width="64" alt="Combined Image" />


###Data Set Summary & Exploration

We begin the project by loadind the data, and obtaining a quick summary of the dimensions of our data set.

* Number of training examples = 34799
* Number of validating examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####Exploratory visualization of the dataset.

In the Jupyter notebook we provide a quick visualization of the type of images in our data set. Namely, we pick a random subset of the labels, and for each of them we pick 10 random images, so that without covering all of the cases, we can get a glimpse of the variotion we will see.

After that, we summarized the data sets by label. Below we show histograms for each of the sets:

Training Set
<img src="examples/Frequencies_training.png" width="240" alt="Combined Image" />

Validation Set
<img src="examples/Frequencies_validation.png" width="240" alt="Combined Image" />

Test Set
<img src="examples/Frequencies_testing.png" width="240" alt="Combined Image" />

We can see that the distribution across labels is very uneven, but at the same time, very similar in all three sets. Since we are interested in the performance on the test and validation sets, we do not need to regularise this distribution. Indeed, if we believe that this is the frequency in which they are found on the German roads, we might be happy to preserve it.
On the other hand, if we had reason to care more about properly classifying certain signs, we would then want to ensure they are properly represented in our sample. 

We still decided to augment the Training data set, which we did by using five transformations. Below we show an example for each of them:

<img src="examples/Transformation_blur.png" width="128" alt="Combined Image" />
<img src="examples/Transformation_homography.png" width="128" alt="Combined Image" />
<img src="examples/Transformation_noise.png" width="128" alt="Combined Image" />
<img src="examples/Transformation_rotation.png" width="128" alt="Combined Image" />
<img src="examples/Transformation_shift.png" width="128" alt="Combined Image" />


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


