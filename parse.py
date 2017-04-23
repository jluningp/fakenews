import csv
import numpy as np

class Parser(object):

    # Creates a Parser object
    # filename: the name of the csv file to be parsed
    # Instantiates the columns, features, and labels fields by
    # parsing the csv
    def __init__(self, filename):
        def make_count(a):
            if a == "":
                return 0
            else:
                return int(float(a))

        def make_bool(a):
            if a == "REAL":
                return 1
            else:
                return 0
            
        with open(filename, 'rt') as feature_file:
            feature_reader = csv.reader(feature_file)

            #separates each row into (label, feature) tuples
            print("Separating features...")
            vector = [(row[1], row[2:]) for row in feature_reader]

            #The column headings (i.e., "title:Trump")
            self.columns = vector[0][1]
            
            vector = vector[1:]
            
            #unzips the tuples to make a label set and a feature set
            print("Unzipping labels...")
            (labels, features) = (np.array([make_bool(a)  for (a, b) in vector]),
                                  np.array([[make_count(x) for x in b] for (a, b) in vector]))

            #All feature, label vectors without the column headings
            self.features = features[1:]
            self.labels = labels[1:]

    # Gets the data split between a train set and a test set
    # Train percentage set to 85% by default (don't now how we should split it, just guessed lol)
    # Returns a dict with "train", "test", and "columns" as its entries. Traind and test are lists
    #         of tuples of (feature vector, label) and columns is the column headings of each
    #         feature vector (i.e., title:trump)
    def get_split_data(self, train_percentage=0.85):
        #check if train_percentage is valid
        if(train_percentage > 1.0 or train_percentage < 0.0):
            print("Invalid train percentage")
            return

        #find the index of the split
        index = int(len(self.labels) * train_percentage)

        #split into train and test, return a dictionary with relevant info
        data = dict()
        data["train"] = (self.features[:index], self.labels[:index])
        data["test"] = (self.features[index:], self.labels[index:])
        data["columns"] = self.columns
        return data
 
