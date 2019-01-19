# Aleena Watson
# CS 445 Portland State University
# This code is formulated for identifying digits in the MNIST
# dataset using 10 perceptrons (identifying digits 0-9). To
# run, you must give the learning rate via commandline, ie:
#
#   python mnist_perceptron.py 0.01

# this allows you to run your experiments in parallel (plug
# your computer into a power source before doing this). I did
# not time my solution, but it takes under 10 minutes to run
# one time.


import sys
import tqdm
import numpy
import torch
import torchvision
import matplotlib.pyplot as plt

test_accuracy = []
train_accuracy = []

# Batch size is a term used in machine learning and refers to
# the number of training examples utilised in one iteration.
# Batch size was not specified in the assignment, so set it to 1.

training_set = torch.utils.data.DataLoader(
    batch_size=1,
    shuffle=True,
    dataset=torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    )
)

test_set = torch.utils.data.DataLoader(
    batch_size=1,
    shuffle=True,
    dataset=torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    )
)

def accuracy_and_confusion_matrix(dataset):
    total = 0.0
    correct = 0.0
    matrix = torch.LongTensor(10, 10).zero_()

    # output is used for "predicted class" in the confusion matrix
    # and target is the "actual class" in the confusion matrix.
    for input, target in tqdm.tqdm(dataset):
        output = (torch.mm(weights, input.view(784, 1)) + biases).argmax()
        total += 1
        if output == target:
            correct += 1
        #end

        # matrix row , column
        matrix[output, target] += 1
    return correct / total, matrix
#end


# batchsize, channels, height, width of the data coming from mnist dataset

# using sys.argv[1] to read learning rate from commandline
# to help speed up completion time
learning_rate = float(sys.argv[1])
epochs = 50

# weights for my perceptrons, 10 perceptrons
# and 784 inputs (typo in project description-785)
# setting my random numbers around the normal
# distribution using 0, and 0.01 gives me the
# prompted starting values of between -0.05, 0.05.
weights = torch.FloatTensor(10, 784).normal_(0, 0.01)

# extra tensor for my bias being a one by ten
# dimensional array, and since the bias is 1,
# we fill it with ones in place.
biases = torch.FloatTensor(10, 1).fill_(1)


print("Accuracy Training set (before training)")
train_accuracy.append(accuracy_and_confusion_matrix(training_set)[0])

print("Accuracy Test set (before training)")
test_accuracy.append(accuracy_and_confusion_matrix(test_set)[0])

for i in range(epochs):
    # for each input we perform the dot product with
    # torch.mm, we reshape the input to be able to do this
    # with .view()
    for input, target in tqdm.tqdm(training_set):
        target = torch.FloatTensor(10).zero_().scatter_(0, target.long(), 1)
        output = (torch.mm(weights, input.view(784,1)) + biases > 0.0).float()
        weights = weights + (learning_rate * (target.view(10,1) - output.view(10,1)) * input.view(1,784))
    #end

    print("Accuracy Training after training #", i)
    train_accuracy.append(accuracy_and_confusion_matrix(training_set)[0])

    print("Accuracy Testing after training #", i)
    test_accuracy.append(accuracy_and_confusion_matrix(test_set)[0])
#end

print("Confusion Matrix (Test Set)")
print(accuracy_and_confusion_matrix(test_set)[1])

print("Learning Rate")
print(learning_rate)

plt.figure()
plt.plot(train_accuracy)
plt.plot(test_accuracy)
lines = plt.plot(test_accuracy, train_accuracy)
plt.setp(lines, color='r', linewidth=2.0)
plt.axis([0, 50, 0, 1])
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()
