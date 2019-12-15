#!/usr/local/bin/python3
###################################

from orient_solver import *
import sys


# Read in training or test data file
#
def read_data(fname, model_type): # Read differently for different versions of model_type
    exemplars = []
    file = open(fname, 'r');
    if model_type == "nearest" or model_type == "none":
        for line in file:
            data = tuple([w.lower() for w in line.split()])
            exemplars += [[data[0], data[1], data[2:]], ] #(image_id , orientation, (pixel data, pixel data ... ))
    elif model_type == "tree":
        with open(fname, "rb") as input:
            return pickle.load(input)
    return exemplars # Exemplars is list of tuples: [(image_id, orientation, (pixel data, pixel data ...)), ... ]


####################
# Main program
#

# Exits program if user does not supply enough input arguments
if len(sys.argv) < 5:
    print("Usage: \n./orient.py [test or train] train_file.txt model_file.txt [model]")
    sys.exit()

# Define user inputs
test_or_train = sys.argv[1] # 'test' or 'train'
model_filename = sys.argv[3] # This is just string "model_file.txt" for now
                           # When training, program should write to this
                           # When testing, program should read data from this file
model_type = sys.argv[4] # 'nearest', 'tree', 'nnet'

# Solver will be its own class containing train and test functions
solver = Solver() # initialize class, written in orient_solver.py


if test_or_train == "train":
    print("Learning model...")
    train_file = sys.argv[2]
    solver.train(train_file, model_type, model_filename) # Should write to model_file.txt differently for each model

if test_or_train == "test":
    print("Loading test data...")

    test_file = sys.argv[2]
    test_data = read_data(test_file, "none")

    print("Testing classifier...")
    file_output = open("output.txt", "w+")
    model_params = read_data(model_filename, model_type)
    accuracy_count = 0

    # Predicts actual orientation for each image in test data set
    for image_vector in test_data: # Each input has only one line of output
        output = solver.solve(image_vector, model_type, model_params)
        if output[1] == image_vector[1]:
            accuracy_count += 1
        # Should print to separate file called output.txt
        file_output.write(str(output[0]) + " " + str(output[1]) + "\n") # should look like " 'image_id' 'orientation' "
    file_output.close()

    accuracy = round(((accuracy_count / len(test_data)) * 100), 2)
    print("model accuracy: ", accuracy)




