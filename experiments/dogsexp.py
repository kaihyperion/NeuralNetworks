from data.dogs import DogsDataset
import numpy as np
from src.models import Dog_Classifier_FC
from data.my_dataset import MyDataset
from src.run_model import run_model
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

model = Dog_Classifier_FC()

dogs_4d = DogsDataset('data/DogSet')

def flatten(images):
    size = (images.size)/images.shape[0]
    size = int(size)
    flat = np.zeros((images.shape[0], size))
    i = 0
    for image in images:
        flatim = image.flatten()
        flat[i] = flatim
        i += 1
    return flat

print('Number of images in Train Partition: ', dogs_4d.trainY.shape[0])
print('Number of images in Valid Partition: ', dogs_4d.validY.shape[0])
print('Number of images in Test Partition: ', dogs_4d.testY.shape[0])
print('Number of Color Channels: ', dogs_4d.trainX.shape[3])
print('Number of dog breeds: ', np.unique(dogs_4d.testY).shape[0])

train_dataset = MyDataset(flatten(dogs_4d.trainX), dogs_4d.trainY)
valid_dataset = MyDataset(flatten(dogs_4d.validX), dogs_4d.validY)
test_dataset = MyDataset(flatten(dogs_4d.testX), dogs_4d.testY)

model, train_loss, train_accuracy = run_model(model, running_mode='train', train_set=train_dataset, valid_set=valid_dataset, batch_size=10, learning_rate=1e-5, n_epochs=100)

print('Number of epochs before terminating: ', len(train_loss['train']))

training_loss = np.asarray(train_loss['train'])
validation_loss = np.asarray(train_loss['valid'])
training_accuracy = np.asarray(train_accuracy['train'])
validation_accuracy = np.asarray(train_accuracy['valid'])

epoch_values = np.arange(1, len(train_loss['train'])+1)

plt.plot(epoch_values, training_loss, label="Training Loss")
plt.plot(epoch_values, validation_loss, label="Validation Loss")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Training/Validation Loss')
plt.title("DogSet Neural Network - Training & Validation Loss")
plt.savefig("DogSet_8a")
plt.cla()

plt.plot(epoch_values, training_accuracy, label="Training Accuracy")
plt.plot(epoch_values, validation_accuracy, label="Validation Accuracy")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Training/Validation Accuracy')
plt.title("DogSet Neural Network - Training & Validation Accuracy")
plt.savefig("DogSet_8b")
plt.cla()

test_loss, test_accuracy = run_model(model, running_mode='test', test_set=test_dataset)
print('Accuracy of Model on Testing Set: ', test_accuracy)