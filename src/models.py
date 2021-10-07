import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.
#
# Note: Avoid using nn.Sequential here, as it prevents the test code from
# correctly checking your model architecture and will cause your code to
# fail the tests.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        # Building Model Architecture use torch.nn.linear
        super(Digit_Classifier, self).__init__()
        self.first_hidden_layer = nn.Linear(784 , 128) #28^2
        self.second_hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)


    def forward(self, inputs):
        # Our hidden layers should have ReLU activation function
        inputs = F.relu(self.first_hidden_layer(inputs))
        inputs = F.relu(self.second_hidden_layer(inputs))
        inputs = self.output_layer(inputs)
        return inputs


class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        self.first_hidden_layer = nn.Linear(12288, 128)
        self.second_hidden_layer = nn.Linear(128,64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, inputs):
        inputs = F.relu(self.first_hidden_layer(inputs))
        inputs = F.relu(self.second_hidden_layer(inputs))
        inputs = self.output_layer(inputs)
        return inputs
        
