################################
# This class has a train function and a function to predict orientations for test data

import shutil
import math
import statistics

class Solver:

    # This function should write to model_file.txt
    # Input arguments may need to be adjusted
    def train(self, data, model, model_filename):
        model_params = open(model_filename, "w+") # Open model_params as write only
                                                  # Write using model_params.write("text")
                                                  # Use model_params.close() once done
        if model == "nearest":
            # data is now original training data:
            # simply copy over model_params
            shutil.copy("train_file.txt", "model_file.txt")

        if model == "tree":
            # Write to model_params
            return "null"
        if model == "nnet":
            # Write to model_params
            return "null"


    def nearest(self, test_image, model_params):
        # Model params is equal to training data
        distance_vector = []
        for training_image in model_params:
            squared_distance = 0
            # Calculate distance from each training image to test image
            for i in range(len(test_image[2])):
                squared_distance += pow((int(test_image[2][i]) - int(training_image[2][i])),2)

            # append distance and classification from training data
            distance = round(math.sqrt(squared_distance),2)
            distance_vector.append((distance, training_image[1]))

        distance_vector.sort() # Sort vector in order of ascending distance
        neighbors = []
        for element in distance_vector[0:50]: # K = 50
            neighbors.append(element[1])# Should append just the orientation value from top 10 nearest neighbors
        return (test_image[0], statistics.mode(neighbors))


    def tree(self, test_image, model_params):

        return "orientation"

    def nnet(self, test_image, model_params):

        return "orientation"


    # Solve function is called by orient.py for each image in the test set
    # Should return the image ID and predicted orientation
    def solve(self, test_image, model, model_params):

        if model == "nearest":
            return self.nearest(test_image, model_params)
        if model == "tree":
            return self.tree(test_image, model_params)
        if model == "nnet":
            return self.nnet(test_image, model_params)

        else:
            print("Unknown model!")

