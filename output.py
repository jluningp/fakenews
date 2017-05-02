import csv
import numpy as np

class ReaderWriter(object):

    # Writes the convolution, hidden, and output weight
    # matrices to filename. Use this to save the weights
    # that the neural net generated.
    @staticmethod
    def write(filename, clayer, hlayer, olayer):
        #helper to write row for layer with index i
        def write_layer(i, layer, csvwriter):
            for line in layer:
                csvwriter.writerow([i] + line.tolist())

        #writes the layers
        with open(filename, 'wt') as csvfile:
            csvwriter = csv.writer(csvfile)
            print("Writing first hidden layer  weights...")
            write_layer(0, clayer, csvwriter)

            print("Writing second hidden layer weights...")
            write_layer(1, hlayer, csvwriter)

            print("Writing output layer weights...")
            write_layer(2, olayer, csvwriter)

            
    # Reads the weight matrices from filename. Use this
    # to load the saved weights.
    # Returns a dict containing "convolution", "hidden",
    # and "output" as keys for each of these weight matrices.
    @staticmethod
    def read(filename):
        #helper to assign keys
        def layer_name(i):
            if i == 0:
                return "convolution"
            if i == 1:
                return "hidden"
            else:
                return "output"

        #loads the weight matrices into layer_dict
        with open(filename, 'rt') as csvfile:
            csvreader = csv.reader(csvfile)

            layers = [[], [], []]
            j = -1
            for row in csvreader:
                #for printing purposes
                i = int(row[0])
                if i != j:
                    print("Loading " + layer_name(i) + " layer weights...")
                    j = i
                
                #for copying layers
                layers[i] = layers[i] + [np.array([float(x) for x in row[1:]])]

            layers = [x for x in layers if len(x) > 0]
            layer_dict = dict()
            for i in range(0, len(layers)):
                layer_dict[layer_name(i)] = np.array(layers[i])

            return layer_dict
