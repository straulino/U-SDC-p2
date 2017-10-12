***Traffic Sign Recognition*** 

In this project we build a Convolution Network and we train it to classify German traffic signs. The aim is to obtain at least 93% accuracy on the validation set, test it on a test set and finally test it on new traffic signs to be obtained from the web.

Below we will give o brief overview of the process we followed to do so, and try to explain why we make the choices we made, as well as where could we potentially improve our Neural Network.

**Data Set Summary & Exploration**

We begin the project by loadind the data, and obtaining a quick summary of the dimensions of our data set.

* Number of training examples = 34799
* Number of validating examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

*Exploratory visualization of the dataset.*

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

<img src="examples/Transformation_blur.png" width="128" alt="Combined Image" /> <img src="examples/Transformation_homography.png" width="128" alt="Combined Image" />
<img src="examples/Transformation_noise.png" width="128" alt="Combined Image" />
<img src="examples/Transformation_rotation.png" width="128" alt="Combined Image" />
<img src="examples/Transformation_shift.png" width="128" alt="Combined Image" />

This transformations should help us obtain a more robust model than we would without them. Small perturbations of the image, that might be due to a number of causes, should have no effect on our classifier. So if we later encounter signs that seem slightly rotated, blurry, further up or down on the image, we should still be able to classify them.

**Design and Test a Model Architecture**

1. Preprocessing the Data

Following the paper http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf, we decided to convert to grayscale, and furthermore we applied histogram equalization (which enlarges the range of the image as a function) to improve the contrast, and we normalized the resulting image to the range [ 0, 1].


2. Model architecture

Again, following the paper above, we decided to use a Convolution Net in which all of the convolution layers feed into a fully connected layer, the idea behind it being that each convolution layer picks up features at different scales, and all scales could be useful for the final classification.

We have three convolution layers followed by two fully connected ones. After each convolution layer we also apply some pooling and dropout, which should help avoid overfitting.

My final model consisted of the following layers:

1 - Convolution layer using a 5 x 5 filter, followed by SELU activation, 2 x 2 x 2 pooling and dropout.
2 - Convolution layer using a 5 x 5 filter, followed by SELU activation, 2 x 2 x 2 pooling and dropout.
3 - Convolution layer using a 5 x 5 filter, followed by SELU activation, 2 x 2 x 2 pooling and dropout.
4 - Fully connected layer, followed by SELU activation and dropout.
5 - Fully connected layer, followed by SELU activation and dropout.


3. Training

We trained the model over 60 Epochs, reducing the learning rate and keep probability every 15 Epochs. We used the AdamOptimizer that we inherited from the LeNet algorithm, and since we obtained good results with this setup, we did not pursue any changes to it.

4. Approach

We began by using the default architecture of the LeNet lab. Since we converted to grayscale as part of our preprocessing, we didn't need to change the input size, and we only needed to adapt the output. The initial result was not very encouraging (around 86%), so we knew we needed to improve all along the pipeline.

Following the Semarnet & LeCunn paper, we added a third convolution layer, and ensured that all three layers fed into the first fully connected layer. We also changed the activation to SELU. These changes improved our validation accuracy to around 93%.

The next step was to augment the data with the transformations we introduced above. This changes improved our validation accuracy to about 94%.

Finally, we introduced dropout, which allowed us to obtain 98.7% accuracy on the validation set.

When we tested it on the test set we got a respectable 95.8%. We believe a few changes to the architecture and a more aggresive dropout might improve our results.

**Test a Model on New Images**

1. We picked 10 new traffic signs from the internet

<img src="examples/newImage0.jpg" width="64" alt="Combined Image" /> <img src="examples/newImage1.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage2.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage3.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage4.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage5.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage6.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage7.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage8.jpg" width="64" alt="Combined Image" />
<img src="examples/newImage9.jpg" width="64" alt="Combined Image" />

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


