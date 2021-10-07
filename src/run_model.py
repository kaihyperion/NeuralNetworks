import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None,
    batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
    """
    This function either trains or evaluates a model.

    training mode: the model is trained and evaluated on a validation set, if provided.
                   If no validation set is provided, the training is performed for a fixed
                   number of epochs.
                   Otherwise, the model should be evaluted on the validation set
                   at the end of each epoch and the training should be stopped based on one
                   of these two conditions (whichever happens first):
                   1. The validation loss stops improving.
                   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs:

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
    learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model
    loss: dictionary with keys 'train' and 'valid'
          The value of each key is a list of loss values. Each loss value is the average
          of training/validation loss over one epoch.
          If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
         The value of each key is a list of accuracies (percentage of correctly classified
         samples in the dataset). Each accuracy value is the average of training/validation
         accuracies over one epoch.
         If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set.
    accuracy: percentage of correctly classified samples in the testing set.

    Summary of the operations this function should perform:
    1. Use the DataLoader class to generate trainin, validation, or test data loaders
    2. In the training mode:
       - define an optimizer (we use SGD in this homework)
       - call the train function (see below) for a number of epochs untill a stopping
         criterion is met
       - call the test function (see below) with the validation data loader at each epoch
         if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results

    """

    # Dictionary with key train and valid
    loss = {'train':[],'valid':[]}
    acc = {'train':[],'valid':[]}



    #Use the data loader class to generate train, validation, etc

    if (running_mode == 'train'):
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        #In the training mode, we have to define OPTIMIZER
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        #call the train function for a number of epoches until a stopping criterion is met
        # this will require a for loop for all the epochs

        #This is for checking the validation because we will need a dataloader for validation
        if valid_set:
            valid_loader = DataLoader(valid_set, shuffle=shuffle)

        for epoch in range(n_epochs):
            #train the function
            model, train_loss, train_acc = _train(model, train_loader, optimizer)
            #store the loss and acc into the dictionary for train lists we've created
            loss['train'].append(train_loss)
            acc['train'].append(train_acc)

            #If validation set is provided:
            if valid_set:
                # call func _test to w/ validation dataloader
                vl,va = _test(model, valid_loader)
                #assign them into appropriate dictionary for valid lists
                loss['valid'].append(vl)
                acc['valid'].append(va)
                if len(loss['valid']) > 1:
                    # check if the valid loss right efore and the currently measure vl difference is less than stop_thr
                    if loss['valid'][-2] - vl < stop_thr:
                        break
        return model, loss, acc

# step 3 if it is in testing mode
    elif running_mode == 'test':
        testing_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
        loss = []
        acc = []
        loss, acc = _test(model, testing_loader)
        return loss, acc

def _train(model,data_loader,optimizer,device=torch.device('cpu')):

    """
    This function implements ONE EPOCH of training a neural network on a given dataset.
    Example: training the Digit_Classifier on the MNIST dataset
    Use nn.CrossEntropyLoss() for the loss function


    Inputs:
    model: the neural network to be trained
    data_loader: for loading the netowrk input and targets from the training dataset
    optimizer: the optimiztion method, e.g., SGD
    device: we run everything on CPU in this homework

    Outputs:
    model: the trained model
    train_loss: average loss value on the entire training dataset
    train_accuracy: average accuracy on the entire training dataset
    """
    # we have to return train loss and accuracy. both AVG.
    #criterion = nn.MSELoss() from pytorch documentation
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct= 0
    total = 0

    for i, data in enumerate(data_loader):
        # get the inputsl data is a list of [inputs, labels]
        inputs, labels = data

        #zero the parameter gradients
        optimizer.zero_grad()

        # forward plus backward plus + optimize
        #outputs = net(inputs)
        outputs = model.forward(inputs.float())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(data_loader)
    train_acc = 100*correct/total

    return model, train_loss, train_acc







def _test(model, data_loader, device=torch.device('cpu')):
    """
    This function evaluates a trained neural network on a validation set
    or a testing set.
    Use nn.CrossEntropyLoss() for the loss function

    Inputs:
    model: trained neural network
    data_loader: for loading the netowrk input and targets from the validation or testing dataset
    device: we run everything on CPU in this homework

    Output:
    test_loss: average loss value on the entire validation or testing dataset
    test_accuracy: percentage of correctly classified samples in the validation or testing dataset
    """

    criterion = nn.CrossEntropyLoss(reduction='sum')
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data

            inputs = inputs.float()
            labels = labels.long()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

    test_loss = running_loss/total
    test_acc = 100*correct/total
    return test_loss, test_acc
