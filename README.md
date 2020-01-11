# K Nearest Neighbors

This algorithm is meant to loop through the training data for each test image, calculate the “distance” between each training image and the test image, select the “k” nearest neighbors, and finally return the classification from the nearest neighbors with the highest probability. 

For this example, the distance was calculated at each iteration by taking the euclidean distance between the pixel data vectors of both the test image and training image. The “k” value used for this example was 50.

How the program works:

When the training function is called for the model “nearest neighbors,” the training function in the program ‘orient_solver.py’ copies over the entire training data set to the file called ‘model_file.txt’. 

When the test function is called for the “nearest neighbors” model from ‘orient.py’, it iterates through each test image in the test data set. For each test image, the solve function is called from ‘orient_solver.py’. This function is passed the test image and ‘model_file.txt’. In the solve function, for each image in the training set, the euclidean distance from the test image is calculated and stored in ‘distance_vector’. The distance is stored along with the orientation value. This vector is then sorted, so that the vector contains the tuples (‘distance’ , ‘orientation’), in order of ascending distance. Next, the first 50 elements of the ‘distance_vector’ are chosen to be the “nearest neighbors”. Out of these “nearest neighbors,” the mode of the orientation value of taken. The test image is finally classified with this orientation value.


# Decision Trees

To implement the image orientation classifier with decision trees, we use the ID3 decision tree learning algorithm.
This algorithm works by selecting features of a dataset which are most discriminative, i.e. provide highest information gain,
for the particular categorization task. 

ID3 recursively builds a decision tree by selecting attributes whose resulting decision boundaries partition the 
data most effectively -- i.e. that minimizes the uncertainty about which of the 4 categories of image orientation is 
present in the test image. These attributes are selected from a set, which in this implementation is the full set of 
pairwise pixel comparisons. The criterion is whether the intensity of pixel A greater than pixel B. Rather than choosing 
from the full 192^2 possible pixel comparisons, we take a random subsample of N attributes (or pixel-pixel comparisons),and 
this number N determines the resulting depth of the tree. We found the most effective number N to be roughly 11 attributes. 
When we tested on really small trees, we found the most predictive attributes compared more distant points.

We found that this algorithm was very sensitive to overfitting, and that shallower trees preformed much better than
deep trees. Currently the tree depth is limited to 11 attributes (i.e. binary decision tree nodes). This setting results 
in classification accuracy which ranges from 55-69% on the test set.

We tried a few image processing techniques to aid classification performance, such as reducing the images to a single 
color value (averaging over the 3 color pixel values). This reduced the image space down to 64 states, rather than 192. 
We hoped that this form of dimensionality reduction might have helped the algorithm select better features, but found that it
performed worse than the algorithm operating over the original images. In future work we might incorporate more advanced 
image processing techniques, such as convolution, which might extract more informative features, especially as they pertain
to the orientation of an image.  

# Neural Network

In this algorithm we use neural nets which are multi-layer network of neurons to classify the orientation required for each image. We have implemeted 5 layer neural network with 1 input layer, 1 output layer and 3 hidden layers. The neural net uses sigmoid activation function in the training phase and makes final prediction using softmax activation function. We have also implemented l1 & l2 regularization with cross entropy for error optimization and calculation. The working of neural network is as follows:-
- First, the training data is preprocessed by performing normalization of the pixel values and implementing one-hot encoding on the labels. Weights are randomly initialized which will be used in the forward pass.
- Second, this preprocessed data goes through the forward pass where at each layer we perform dot product of the input data with a randomly initialized weight and the result is passed as input for the next hidden layer. Between each layer we perform activation step as this helps the model to make sense of the data. As mentioned we have implemented sigmoid activation since it exists between 0 to 1 and we want to predict the probability for each class which will also lies between (0,1). We also used the Relu activation function but the results that we got using relu were suprisingly poor as compared to the sigmoid activation even though relu is most used and popular activation function.
- Third, output of forward pass is then used for backward pass where we use the sigmoid derivative function to calculate the gradients which are inturn used to calculate the error using cross entropy and regularization. Using this error the weights are updated and this whole process is repeated in each epoch.
- Finally, we use the softmax function the precict the class of the predicted orientation.

For this model we are getting 78-80% accuracy, thus we can say that neural net is the best model. 

The error graph for this model is attached with the files.

## Direction to execute : ./orient.py [test or train] train_file.txt model_file.txt [model]
## Where, model = knn, nnet or tree

