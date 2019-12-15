################################
# This class has a train function and a function to predict orientations for test data

import shutil
import math
import statistics
import numpy as np
import random
import pickle
import neuralNet

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
            result = train_tree(data)
            with open(model_filename, "wb") as out:
                pickle.dump(result, out)

            return "null"
        if model == "nnet" or model == "best":
            netWeight = neuralNet.train(data)
            pickle.dump(netWeight, open(model_filename,"wb"))
            print("Model Ready!")

        model_params.close()


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

        img = np.array([int(x) for x in test_image[2]])

        result = model_params.solve(img)

        return (test_image[0], str(result))


    def nnet(self, test_data, model_params):
        netWeight = pickle.load(open(model_params, "rb"))
        orientation = neuralNet.test(netWeight, test_data)
        return orientation


    # Solve function is called by orient.py for each image in the test set
    # Should return the image ID and predicted orientation
    def solve(self, test_image, model, model_params):

        if model == "nearest":
            return self.nearest(test_image, model_params)
        if model == "tree":
            return self.tree(test_image, model_params)
        if model == "nnet" or model == "best":
            return self.nnet(test_image, model_params)

        else:
            print("Unknown model!")


def read_data(train_file):
    with open(train_file, "r") as input:
        data = []
        for line in input:
            vals = line.split()
            cat = int(vals[1])
            img = [int(v) for v in vals[2:]]
            data.append((vals[0], cat, np.array(img)))
    return data


class TreeNode():
    def __init__(self, attr=None, data=None, terminal=None):
        self.attr = attr
        self.data = data
        self.terminal = terminal
        self.left_child = None
        self.right_child = None

    def solve(self, input):
        curr_node = self
        # input = reduce_colors(input)

        while curr_node.terminal is None:
            d = decision(input, curr_node.attr)
            if d == "l":
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child

        return curr_node.terminal

classes = [0, 90, 180, 270]


def train_tree(train_file, depth=11):

    data = read_data(train_file)
    # data = reduce_data(data)

    img_size = data[0][2].shape[0]

    attributes = [(x, y) for x in range(img_size)
                            for y in range(img_size)]

    random.shuffle(attributes)

    result = id3(data, attributes[:depth])

    return result

def reduce_data(data):
    result = []
    for d in data:
        x = d[2].reshape(-1, 3)
        new_img = np.mean(x, axis=1)
        result.append((d[0], d[1], new_img))

    return result


def id3(data, attributes):

    node = TreeNode(data=data)

    # check if any of the classes are a perfect match
    for c in classes:
        if all(d[1] == c for d in data):
            node.terminal = c
            return node

    # if there's no more attributes to check, set the
    # terminal node to the category with greatest proportion
    if len(attributes) == 0:
        dist = class_dist(data)
        max_class = np.argmax(dist)
        node.terminal = classes[max_class]
        return node

    attrs = []

    for a in attributes:
        S1, S2 = split_on_attr(data, a)

        # compute information gain for this binary branch
        gain = info_gain(S1, S2)

        # keep track of branch and info gain value
        attrs.append((a, gain))

    # find the highest information gain attribute for the root node
    attrs = sorted(attrs, key=lambda x: x[1], reverse=True)
    best = attrs[0][0]

    # set branch attribute to the one with highest information gain
    node.attr = best

    remaining_attrs = [x[0] for x in attrs[1:]]

    # split using the best (i.e. highest infogain) attribute
    S1, S2 = split_on_attr(data, best)

    for i, v in enumerate([S1, S2]):
        if len(v) == 0:
            dist = class_dist(S1+S2)
            max_class = np.argmax(dist)
            child = TreeNode()
            child.terminal = max_class

            node.left_child = child
            node.right_child = child

            break
        else:
            child = id3(v, remaining_attrs)
            if i == 0:
                node.left_child = child
            else:
                node.right_child = child

    return node

def split_on_attr(data, attr):
    decisions = []
    for sample in data:
        dec = decision(sample[2], attr)
        decisions.append((dec, sample))

    # make subsets of the examples using the binary decision
    S1 = [d[1] for d in decisions if d[0] == 'l']
    S2 = [d[1] for d in decisions if d[0] == 'r']

    return S1, S2


def info_gain(d1, d2):
    ent_both = entropy(d1+d2)
    ent_1 = entropy(d1)
    ent_2 = entropy(d2)

    gain = ent_both - sum([len(d) / len(d1+d2) * ent
                           for d, ent in [(d1, ent_1),
                                          (d2, ent_2)]])

    return gain


def reduce_colors(img):
    x = img.reshape(-1, 3)
    new_img = np.mean(x, axis=1)
    return new_img


def decision(img, attr, method="pairwise"):
    if method == "pairwise":
        if img[attr[0]] < img[attr[1]]:
            return 'l'
        return 'r'


def class_dist(data, num_bins=4):
    dist = np.ones(shape=num_bins)

    for d in data:
        if d[1] == 0:
            dist[0] += 1
        elif d[1] == 90:
            dist[1] += 1
        elif d[1] == 180:
            dist[2] += 1
        elif d[1] == 270:
            dist[3] += 1

    return dist


def entropy(data):
    dist = class_dist(data)
    dist = dist / sum(dist)
    ent = -sum([p * np.log2(p) for p in dist])
    return ent
