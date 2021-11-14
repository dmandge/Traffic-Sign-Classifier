# Traffic-Sign-Classifier
To build a Traffic Sign Recognition Classifier

## Overview
The goal of this project is to implement concepts of Deep Neural Network and Convolutional Neural Network to image classification and classify German Traffic Signs
This is achieved in following steps:
1. Load German Traffic Sign dataset containing training, validation and test sets
2. Visualize the dataset (I did this by displaying one image from each class and plotting a bar chart showing number of images in each class for all three sets)
3. Design, train and test a model architecture
4. Tune hyper parameters to generate a highly accurate model
5. Use this model to make predictions on new images
6. Analyze softmax probabilities of new images
7. Visualize the Neural Network at its convolutional layer

## Dataset Summary
* Traffic Sign Dataset –
    * The traffic sign data to be used for training model is saved in three pickle files –
      * train.p
      * valid.p
      * test.p
    * Pickled data is a dictionary with 4 values
      * Features - 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels)
      * Labels - 1D array containing the label/class id of the traffic sign
      * Sizes - list containing tuples, (width, height) representing the original width and height the image
      * Coords - list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.
    * The file ‘signnames.csv’ contains traffic sign ids for each class and their name mappings
    * Images in the pickle files are all color images resized to 32X32
* Dataset Summary –
    * I used shape function from Numpy library on ‘features’ and ‘labels’ of pickle dataset to determine number of examples in all three datasets and along with width, height and channels in each image.
      * Number of training examples = 34799
      * Number of validation examples = 4410
      * Number of testing examples = 12630
      * Image data shape = (32, 32, 3)
    * I determined number of unique traffic signs used in the training dataset using the numpy function ‘unique’
      * Number of classes = 43
    * Using readcsv function from pandas library, I mapped traffic sign names of all 43 classes to their IDs
 * Data Visualization –
    * For visualization of the dataset I plotted one random image from each class from training set along with its mapped traffic sign name
    
    ![1](https://user-images.githubusercontent.com/59345845/141694847-4676134e-48c5-4c89-899c-9d1981ca31d6.JPG)
    
    *Figure 1: Example of data visualization*
    
## Design and test a Model Architecture
Training set is used to design and tune a deep learning model to accurately recognize traffic signs from validation and test set as well as traffic signs belonging to one of the 43 defined classes but which are taken from the web and are not in either of the three sets.
* Preprocess the dataset –
    * Grayscale – used to convert three channel data into one channel. This is useful to reducetraining time
    * Normalization – Data is normalized to (-1,1) so that it has approximately zero mean and equal variance. This is done to compress data distribution and simplify training of the dataset using single learning rate
    * Shuffling – It’s important to include this step to avoid the model training to get influenced by sequence of the images in the dataset
* Model Architecture – I used LeNet architecture described in the classroom with some modifications as described below.

    ![lenet](https://user-images.githubusercontent.com/59345845/141694994-aed36898-f69f-46aa-9bee-33b49a6ad6f4.JPG)
    Figure 3: Original LeNet Model Architecture
    
    Layers in LeNet: 
* Layer 1:
  * Convolutional. Input = 32x32x1. Output = 28x28x6, Filter = 5x5, Stride = 1x1
  * Activation – RELU
  * Pooling, Input = 28x28x6. Output = 14x14x6, kernel = 2x2, Stride = 2x2
* Layer 2: 
  * Convolutional. Input = 14x14x6 Output = 10x10x16, Filter = 5x5, Stride = 1x1
  * Activation – RELU
  * Pooling. Input = 10x10x16. Output = 5x5x16, kernel = 2x2, Stride = 2x2
* Layer 3: 
  * Convolutional. Input = 5x5x16 Output = 1x1x400, Filter = 5x5, Stride = 1x1
  * Activation – RELU
  * Flatten. Input = 1x1x400. Output = 400.
  * Dropout
* Layer 4: 
  * Fully Connected. Input = 400. Output = 120 (sigma = 0.1)
  * Activation – RELU
* Layer 5: 
  * Fully Connected. Input = 120. Output = 84 (sigma = 0.1)
  * Activation – RELU
* Layer 6: 
  * Fully Connected. Input = 84. Output = 43 (sigma = 0.1)
  
Hyper-parameters for training of the model –
1. Optimizer – Adam Optimizer from LeNet lab in the classroom.
2. Batch size – 128
3. Epochs - 40
4. Learning rate – 0.001
5. Mu – 0
6. Sigma – 0.1
7. Dropout keep probability = 0.2

Training pipeline
* First I set up the tensorflow placeholder variables – x to hold input batch of image dimension 32x32x1and y to store labels. This labels are integers which are then one hot encoded
* Next, I passed the input data x to from the LeNet function as descried above to calculate logits
* These logits are then compared with one hot encoded label values using tf.nn.softmax_cross_entropy_with_logits() functions and calculate cross entropy
to measure difference between logits and ground truth labels.
* Tf.reduce() mean function is used to average this cross entropy
* This is passed through Adam Optimizer to minimize the loss function similar to Stochastic gradient descent
* Finally, I used the minimize() function on the optimizer to backpropogate the training network and minimize training loss

Evaluation Pipeline
* Once model is trained, it is passed through this pipeline to evaluate its accuracy.
* This model measures whether a given prediction is correct by comparing logits with one hot encoded y labels
* Then we calculate model’s overall accuracy by taking a mean of individual prediction accuracy
* This is done by dividing dataset into batches
* At the time of model training, batches are passed through network for every epoch and the end of every epoch we calculate validation accuracy.
* Once the model is completely trained, it is saved to be used later on test dataset

Final Accuracy
* ***Training set – 0.999***
* ***Validation set – 0.968***
* ***Test set – 0.953***

## Model Test on new images
Five German signs taken off the internet to test new images source: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

![Capture](https://user-images.githubusercontent.com/59345845/141695320-c46ed192-0eb3-46ce-956a-2aa872d58bf6.JPG)

*Figure: Test images after preprocessing*

***The model was successfully able to predict all five images which gives an accuracy of 100%, higher than test set accuracy of 95.3%***

### Visualizing the Neural Network
Following figure shows an output of the first convolutional layer of a fully trained model in for first test image of – speed limit 30kph. From the feature maps, it looks like images boundaries and contrast is highlighted more than other aspects of the image, to distinguish the image. This is more apparent in the first two layers.

![Capture2](https://user-images.githubusercontent.com/59345845/141695395-507dab68-0544-4ec4-9798-3954a14b89ec.JPG)


    
