import numpy as np
import torch
from src.models import Digit_Classifier
from data.load_data import load_mnist_data
from data.my_dataset import MyDataset
from src.run_model import run_model
import time
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

model = Digit_Classifier()

sizes = [500, 1000, 1500, 2000]
sizes_np = np.asarray(sizes)
tr_feat_np = []
tr_tar_np = []

_, test_features, _, test_targets = load_mnist_data(10, 0.0, 100)
test_dataset = MyDataset(test_features, test_targets)

for size in sizes:
    train_features, _, train_targets, _ = load_mnist_data(10, 1.0, (size/10))
    tr_feat_np.append(train_features)
    tr_tar_np.append(train_targets)

tr_datasets = []

for i in range(len(sizes)):
    train_dataset = MyDataset(tr_feat_np[i], tr_tar_np[i])
    tr_datasets.append(train_dataset)

models = []
losses = []
accs = []

train_times = np.zeros(4)
class_acc = np.zeros(4)

for i in range(len(sizes)):
    start = time.time()
    model, train_loss, train_acc = run_model(model, running_mode='train', train_set=tr_datasets[i], batch_size=10, n_epochs=100)
    end = time.time()
    train_time = end - start
    train_times[i] = train_time
    print('Time to train size = {}:'.format(sizes[i]), train_time)
    models.append(model)
    losses.append(train_loss)
    accs.append(train_acc)

    test_loss, test_accuracy = run_model(model, running_mode='test', test_set= test_dataset)
    class_acc[i] = test_accuracy



plt.plot(sizes_np, train_times)
plt.xlabel('Number of Training Examples')
plt.ylabel('Amount of Training Time')
plt.title("Mnist Neural Network - Training Examples and Time")
plt.savefig("MNIST_1")
plt.cla()

plt.plot(sizes_np, class_acc)
plt.xlabel('Number of Training Examples')
plt.ylabel('Classification Accuracy')
plt.title("Mnist Neural Network - Classification Accuracy on Test Set")
plt.savefig("MNIST_3")
plt.cla()