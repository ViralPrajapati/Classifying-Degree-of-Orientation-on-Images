# vkprajap-ngrover-aamatuni-a4
a4 created for vkprajap-ngrover-aamatuni

K Nearest Neighbors

This algorithm is meant to loop through the training data for each test image, calculate the “distance” between each training image and the test image, select the “k” nearest neighbors, and finally return the classification from the nearest neighbors with the highest probability. 

For this example, the distance was calculated at each iteration by taking the euclidean distance between the pixel data vectors of both the test image and training image. The “k” value used for this example was 50.

How the program works:

When the training function is called for the model “nearest neighbors,” the training function in the program ‘orient_solver.py’ copies over the entire training data set to the file called ‘model_file.txt’. 

When the test function is called for the “nearest neighbors” model from ‘orient.py’, it iterates through each test image in the test data set. For each test image, the solve function is called from ‘orient_solver.py’. This function is passed the test image and ‘model_file.txt’. In the solve function, for each image in the training set, the euclidean distance from the test image is calculated and stored in ‘distance_vector’. The distance is stored along with the orientation value. This vector is then sorted, so that the vector contains the tuples (‘distance’ , ‘orientation’), in order of ascending distance. Next, the first 50 elements of the ‘distance_vector’ are chosen to be the “nearest neighbors”. Out of these “nearest neighbors,” the mode of the orientation value of taken. The test image is finally classified with this orientation value.
