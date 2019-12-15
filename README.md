# vkprajap-ngrover-aamatuni-a4
a4 created for vkprajap-ngrover-aamatuni

K Nearest Neighbors

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

We found that this algorithm was very sensitive to overfitting, and that shallower trees preformed much better than
deep trees. Currently the tree depth is limited to 11 attributes (i.e. binary decision tree nodes).

We tried a few image processing techniques to aid classification performance, such as reducing the images to a single 
color value (averaging over the 3 color pixel values). This reduced the image space down to 64 states, rather than 192. 
We hoped that this form of dimensionality reduction might have helped the algorithm select better features, but found that it
performed worse than the algorithm operating over the original images. In future work we might incorporate more advanced 
image processing techniques, such as convolution, which might extract more informative features, especially as they pertain
to the orientation of an image.  