# ----------------------------------------------------------------------------------------------------------------------
# NOTE: excessive in-line comments were just for my own learning purposes
# ----------------------------------------------------------------------------------------------------------------------

# Importing necessary libraries
import torch
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm                                                                       # For progress bar
import matplotlib.pyplot as plt                                                             # For plotting
from sklearn.model_selection import train_test_split                                        # For easy data splitting


# Defining function to convert data into usable Tensor of floats within [0, 1]
def normalize(x_denormalized, y_denormalized):
    x_denormalized = x_denormalized / np.float32(255)                                       # Converting data to floats within [0, 1]
    x_denormalized = x_denormalized[..., None]                                              # Adding new dimension for compatibility
    x_denormalized = x_denormalized.transpose([0, 3, 1, 2])                                 # Rearranging dimensions for compatibility
    x_normalized = torch.from_numpy(x_denormalized)                                         # Converting data to Tensor
    y_normalized = torch.from_numpy(y_denormalized).view(-1)                                # Converting labels to Tensor
    return x_normalized, y_normalized


# Defining function to train neural network over single epoch
def train(train_dataset):
    neural_network.train()                                                                  # Setting model to training mode
    total_accumulated_loss = 0                                                              # Defining variable to accumulate loss over current epoch
    proper_prediction_count = 0                                                             # Defining variable to accumulate count of successfully predicted labels over current epoch
    for x, y in train_dataset:                                                              # For each batch of samples in data set
        x, y = x.cuda(), y.cuda()                                                           # Boosting computation power
        optimizer.zero_grad()                                                               # Clearing gradient since backward() accumulates gradient already
        y_predicted = neural_network(x)                                                     # Computing predictions of current model
        loss = loss_function(y_predicted, y)                                                # Finding loss between predictions and actual labels
        total_accumulated_loss += loss.item()                                               # Accumulating loss between predictions and actual labels
        loss.backward()                                                                     # Computing the gradient of the loss WRT each model parameter
        optimizer.step()                                                                    # Updating model parameters at learning rate according to direction of steepest descent
        batch_winning_labels = y_predicted.argmax(dim=1, keepdim=True)                      # Extracting the predicted winning labels (y_pred is 32x10 ie. 32 examples each with weights for labels 0-9)
        batch_proper_labels = y.view_as(batch_winning_labels)                               # Extracting the true winning labels
        proper_prediction_count += batch_winning_labels.eq(batch_proper_labels).sum().item()  # Updating count of successful predictions by the number of matching true and predicted labels in current batch
    accumulated_loss = total_accumulated_loss / len(train_dataset.dataset)                  # Computing average loss per sample over current epoch
    percentage_correct = proper_prediction_count / len(train_dataset.dataset)               # Computing percentage of correct predictions over current epoch
    return accumulated_loss, percentage_correct


# Defining function to test neural network
def test(test_dataset):
    neural_network.eval()                                                                   # Setting model to testing mode
    total_accumulated_loss = 0                                                              # Defining variable to accumulate loss over test
    proper_prediction_count = 0                                                             # Defining variable to accumulate count of successfully predicted labels over test
    with torch.no_grad():                                                                   # Turning off gradient since not used during testing (only during training)
        for x, y in test_dataset:                                                           # For each batch of samples in data set
            x, y = x.cuda(), y.cuda()                                                       # Boosting computation power
            y_predicted = neural_network(x)                                                 # Computing predictions of final model
            total_accumulated_loss += loss_function(y_predicted, y).item()                  # Accumulating loss between predictions and actual labels
            batch_winning_labels = y_predicted.argmax(dim=1, keepdim=True)                  # Extracting the predicted winning labels
            batch_proper_labels = y.view_as(batch_winning_labels)                           # Extracting the true winning labels
            proper_prediction_count += batch_winning_labels.eq(batch_proper_labels).sum().item()  # Updating count of successful predictions by the number of matching true and predicted labels in current batch
    accumulated_loss = total_accumulated_loss / len(test_dataset.dataset)                   # Computing average loss per sample over test
    percentage_correct = proper_prediction_count / len(test_dataset.dataset)                # Computing percentage of correct predictions over test
    return accumulated_loss, percentage_correct


# Defining neural network class
class Net(torch.nn.Module):
    def __init__(self, output_size, input_size):
        super(Net, self).__init__()
        self.convolution1 = torch.nn.Conv2d(input_size[0], 16, 3, padding=1)                # Defining first convolution layer with 28 inputs, 16 outputs, filter size of 3, and padding of 1
        self.convolution2 = torch.nn.Conv2d(16, 32, 3, padding=1)                           # Defining second convolution layer with 16 inputs, 32 outputs, filter size of 3, and padding of 1
        self.linear3 = torch.nn.Linear(32 * ((input_size[1] // 4) ** 2), 512)               # Defining third linear layer with 1568 (32x7x7) inputs and 512 outputs
        self.linear4 = torch.nn.Linear(512, 256)                                            # Defining fourth linear layer with 512 inputs and 256 outputs
        self.linear5 = torch.nn.Linear(256, output_size)                                    # Defining fifth linear layer with 1568 (32x7x7) inputs and 512 outputs
        self.pooling = torch.nn.MaxPool2d(2)                                                # Defining pooling layer to emphasize features (data retains size)
        self.relu = torch.nn.ReLU()                                                         # Defining ReLU activation function
        self.input_size = input_size                                                        # Retaining size for reshaping data

    def forward(self, x):                                                                   # Defining data flow through neural network
        layer_1 = self.pooling(self.relu(self.convolution1(x)))                             # Passing data through convolution layer 1, then capping with ReLU, then max pooling
        layer_2 = self.pooling(self.relu(self.convolution2(layer_1)))                       # Passing data through convolution layer 2, then capping with ReLU, then max pooling
        layer_2_flattened = layer_2.view(-1, 32 * ((self.input_size[1] // 4) ** 2))         # Flattening data
        layer_3 = self.relu(self.linear3(layer_2_flattened))                                # Passing data through linear layer 3, then capping with ReLU function
        layer_4 = self.relu(self.linear4(layer_3))                                          # Passing data through linear layer 4, then capping with ReLU function
        layer_5 = self.linear5(layer_4)                                                     # Passing data through linear layer 5 (no capping since loss function need unnormalized input)
        return layer_5


# Defining training constants
EPOCH_NUMBER = 5                                                                            # Number of passes over entire training dataset
BATCH_SIZE = 32                                                                             # Number of samples loaded per batch (for performance enhancement)

# Importing 28x28 grayscale images from 10 classes of hand written digits (60k for training, 10k for validation/testing)
training_dataset = datasets.MNIST(root='./data', train=True, download=True)
testing_dataset = datasets.MNIST(root='./data', train=False, download=True)

# Extracting relevant information from dataset
x_training = training_dataset.data.numpy()
y_training = training_dataset.targets.numpy()
x_testing = testing_dataset.data.numpy()
y_testing = testing_dataset.targets.numpy()

# Evenly splitting testing data into validation set and test set (validation set used to intermediately evaluate "test" performance throughout learning)
x_validation, x_testing, y_validation, y_testing = train_test_split(x_testing, y_testing, test_size=0.5, random_state=1)

# Normalizing data into usable Tensor of floats within [0, 1]
x_training, y_training = normalize(x_training, y_training)
x_validation, y_validation = normalize(x_validation, y_validation)
x_testing, y_testing = normalize(x_testing, y_testing)

# Creating dataset
training_dataset = torch.utils.data.TensorDataset(x_training, y_training)
validation_dataset = torch.utils.data.TensorDataset(x_validation, y_validation)
testing_dataset = torch.utils.data.TensorDataset(x_testing, y_testing)

# Loading dataset (batching is for processing efficiency, shuffling is to avoid recurring local minima)
training_dataset = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
testing_dataset = torch.utils.data.DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Extracting key data set characteristics
image_pixels = np.prod(x_training.shape[1:])
number_of_labels = int(y_training.max() - y_training.min() + 1)

# Creating neural network
neural_network = Net(number_of_labels, x_training.shape[1:])                                # Configuring model with 784 image pixels as input and and 10 label classes as output
neural_network.cuda()                                                                       # Boosting computation power

# Defining loss (error) function
loss_function = torch.nn.CrossEntropyLoss()                                                 # Quantifying difference between predicted label and actual label  (does not need normalized softmax y_pred input)

# Defining parameter updating algorithm
optimizer = torch.optim.Adam(neural_network.parameters())                                   # Adam is alternative to classic gradient descent

# Defining empty containers to track losses and accuracy over training/validation
loss_history = []
accuracy_history = []

# Training the neural network
for epoch in tqdm(range(EPOCH_NUMBER)):                                                     # For each pass over the full data set (tqdm for runtime progression bar display)
    training_loss, training_accuracy = train(training_dataset)                              # Training the neural network
    validation_loss, validation_accuracy = test(validation_dataset)                         # Testing the neural network
    loss_history.append((training_loss, validation_loss))                                   # Tracking training/validation loss over current epoch
    accuracy_history.append((training_accuracy, validation_accuracy))                       # Tracking training/validation accuracy over current epoch

# Plotting training/validation losses
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.plot(range(len(loss_history)), np.array(loss_history)[:, 0])                            # Plotting training loss
plt.plot(range(len(loss_history)), np.array(loss_history)[:, 1])                            # Plotting validation loss
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['Training Loss', 'Validation Loss'])

# Plotting training/validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(len(loss_history)), np.array(accuracy_history)[:, 0])                        # Plotting training accuracy
plt.plot(range(len(loss_history)), np.array(accuracy_history)[:, 1])                        # Plotting validation accuracy
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()

# Testing test data
testing_loss, testing_accuracy = test(testing_dataset)

# Printing testing results
print(f"\n\nTesting Loss: {testing_loss}\nTesting Accuracy: {testing_accuracy}")
