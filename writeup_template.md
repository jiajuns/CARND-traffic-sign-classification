#**Traffic Sign Recognition** 

##Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "gaussian blur"
[image3]: ./examples/random_noise.jpg "median blur"

[image4]: ./examples/loss_curve.png "loss curve"

[image5]: ./examples/no_entry.jpg "no_entry"
[image6]: ./examples/right_turn.jpg "right_turn"
[image7]: ./examples/children_crossing.jpg "children_crossing"
[image8]: ./examples/stop.jpg "stop"
[image9]: ./examples/no_truck_passing.jpg "no_truck_passing"

---
### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the many samples per traffic sign. It is found the ratio of different traffic sign is consistent across training, validation and testing set. However, the dataset is not quite balance.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Data preprocessing
I decided not to convert the images to grayscale because color in the traffic sign do convey a lot of meaning.
For example, a sign in red or blue are completely different.

As the first step, I normalized the image data because I found images have different brightness.

I decided to generate additional data as mentioned in data exploration the training dataset is not balance. To add more data, I use gaussian blurring and median blurring to augement traffic sign that has less than 500 image in the training set.

Here is an example of an original image and an augmented image:

![gaussian blur][image2]
![median blur][image3]


#### 2. Model architecture

My final model use a architecture that is similiar to Alexnet:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 7x7x64 size, 1x1 stride, valid padding        |
| RELU					|												|
| Max pooling	      	| 2x2 stride                    				|
| Convolution 3x3     	| 5x5x96 size, 1x1 stride, valid padding        |
| RELU					|												|
| Max pooling	      	| 2x2 stride                    				|
| Convolution 3x3     	| 3x3x128 size, 1x1 stride, valid padding       |
| RELU					|												|
| Max pooling	      	| 2x2 stride                    				|
| Fully connected		| 21600 input, 2048 output       				|
| Fully connected		| 2048 input, 1024 output 						|
| Fully connected		| 1024 input, 43 output     					|
| Softmax				| etc.        									|
|						|												|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam optimizor, batch size is 64, 20 epoch and 1e-3 as initial learning rate. I also use expotential learning rate decay.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.1\%
* validation set accuracy of 98.4\%
* test set accuracy of 96.6\%

The training and validation loss is shown below:
![loss curve][image4]

If a well known architecture was chosen:
* I choose AlexNet architecture
* AlexNet is the first successful CNN applied on image net challenge. Since ImageNet has way more classes than traffic sign. I think AlexNet will capable to capture the variability of traffic sign.
* When choosing dropout rate 0.5, the training accuracy and validation accuracy is very close. This implies this model do generalize to the unseen data and I did not see any overfitting happening.
 
###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![no entry][image5] ![right turn][image6] ![children crossing][image7] 
![stop sign][image8] ![no truck passing][image9]

The third image might be difficult to classify because children crossing does not have many samples in the training set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| no entry      		| no entry   									| 
| right turn   			| right turn									|
| children crossing		| Right-of-way at the next intersection			|
| stop sign	      		| stop sign 					 				|
| no truck passing   	| no truck passing    							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For the first image, the model is relatively sure that this is a no entry sign (probability of 0.985), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .985         			| Stop sign   									| 
| .013    				| No passing									|
| .001					| Yield											|
| .000	      			| Turn left ahead				 				|
| .000				    | speed limit 30 km/h  							|


For the first image, the model is relatively sure that this is a Turn right sign (probability of 0.99), and the image does contain a turn right. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Turn right   									| 
| .000    				| Ahead only 									|
| .000					| Turn left										|
| .000	      			| Keep left 					 				|
| .000				    | Roundabout mandatory 							|

For the first image, the model is not sure that this is a children passing sign (probability of 0.546), and the image does not contain a children passing. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .546         			| Right-of-way at the next intersection			| 
| .392     				| End of passing								|
| .011					| End of all speed and passing limits			|
| .009	      			| Roundabout mandatory			 				|
| .007				    | General caution     							|

For the first image, the model is not sure that this is a stop sign (probability of 0.497) but it is relative sure than any other class, and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .497         			| Stop sign   									| 
| .099     				| No entry 										|
| .036					| Bumpy road									|
| .024	      			| Keep right					 				|
| .021				    | Traffic signals      							|

For the first image, the model is relatively sure that this is a no passing for truck sign (probability of 0.985), and the image does contain a no passing for truck sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .985         			| No passing for vehicles over 3.5 metric tons	| 
| .012     				| End of no passing by vehicles over 3.5 metric tons
| .002					| Ahead only									|
| .000	      			| Right-of-way at the next intersection			|
| .000				    | Priority road     							|


