import argparse
from parse import Parser
from xml_parser import ParseXML
from output import ReaderWriter
from neural_net import NeuralNet

class Info(object):
    def __init__(self, datafile, xmlfile, weightfile, percentage, learning_rate, hidden_layers, iterations):
        self.datafile = datafile
        self.xmlfile = xmlfile
        self.weightfile = weightfile
        self.percentage = percentage
        if (not percentage):
            self.percentage = 1.0
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        if iterations:
            self.iterations = int(iterations)
        self.dataset = None
        self.vector = None
        self.weights = None
        self.nn = None
        
    def read_in_dataset(self):
        parser = Parser(self.datafile)
        parser.parse()
        self.dataset = parser.get_split_data(self.percentage)

    def read_in_xml(self):
        self.read_in_dataset()
        parser = ParseXML(self.xmlfile, self.dataset["columns"])
        parser.parse()
        self.vector = parser.make_vector()

    def read_in_weights(self):
        self.weights = ReaderWriter.read(self.weightfile)

    def nn_to_train(self):
        self.read_in_dataset()
        structure = dict()
        structure["num_inputs"] = len(self.dataset["columns"])
        structure["num_hidden"] = int(self.hidden_layers)
        structure["num_outputs"] = 1
        self.nn = NeuralNet(structure, self.learning_rate)
        
    def nn_from_weights(self):
        self.read_in_weights()
        num_inputs = len(self.weights["convolution"]) - 1
        num_outputs = len(self.weights["output"][0])
        num_hidden = len(self.weights["convolution"][0])
        structure = dict()
        structure["num_inputs"] = num_inputs
        structure["num_outputs"] = num_outputs
        structure["num_hidden"] = num_hidden
        nn = NeuralNet(structure, 0.5)
        nn.input_to_hidden = self.weights["convolution"]
        nn.hidden_to_layer = self.weights["hidden"]
        nn.layer_to_output = self.weights["output"]
        self.nn = nn
        
def classify(info):
    info.read_in_xml()
    info.nn_from_weights()

    result = info.nn.forward_propagate(info.vector)
    return result

def train(info):
    print("Initializing neural network...")
    info.nn_to_train()

    print("Training neural network for {} iterations...".format(info.iterations))
    info.nn.train(info.dataset["train"][0], info.dataset["train"][1], iterations=info.iterations)
    
    ReaderWriter.write(info.weightfile, info.nn.input_to_hidden, info.nn.hidden_to_layer, info.nn.layer_to_output)

    return info.nn.test(info.dataset["test"][0], info.dataset["test"][1])
    
def main():
    parser = argparse.ArgumentParser(description="Use a neural network to classify fake news.")
    parser.add_argument("datafile", help="the csv file containing the training data and columns")
    parser.add_argument("weightfile", help="the csv file where weights are/should be stored")
    parser.add_argument("-t", "--train", metavar="train_percentage", type=float, action="store", help="train the neural network")
    parser.add_argument("-c", "--classify", metavar="xml_file", action="store", help="the XML file to be classified")
    parser.add_argument("-l", "--learning-rate", metavar="learning_rate", type=float, action="store", help="the learning rate of the neural network")
    parser.add_argument("-n", "--hidden-layers", metavar="number_of_hidden_layers", type=int, action="store", help="the number of hidden layers")
    parser.add_argument("-i", "--iterations", metavar="iters", action="store", help="the number of iterations to train with")
    args = parser.parse_args()

    if((not args.train) and (not args.classify)):
        print("ERROR: -t or -c flag required")
        print("Type main --help for usage")
        return 1

    if((args.train) and (args.classify)):
        print("ERROR: -t and -c flags cannot be used at the same time")
        print("Type main --help for usage")
        return 1

    if((args.train) and ((not args.learning_rate) or (not args.hidden_layers) or (not args.iterations))):
        print("ERROR: -t, -l, -n, and -i flags must be used together")
        print("Type main --help for usage")
        return 1

    info = Info(args.datafile, args.classify, args.weightfile, args.train, args.learning_rate, args.hidden_layers, args.iterations)
                      
    if(args.train and args.learning_rate and args.hidden_layers and args.iterations):
        accuracy = train(info)
        print("---TRAINING FINISHED---")
        print("Accuracy: {}".format(accuracy))
        return 0

    if(args.classify):
        result = classify(info)
        print("")
        print("----RESULTS---")
        print("Score: {}% real".format(round(result[0] * 100, 2)))
        if result > 0.5:
            print("Judgement: REAL")
        else:
            print("Judgement: FAKE")
        return 0

    return 1
        
if __name__ == "__main__": main()
