# Ron Shlomi
# Artificial Neural Network

# TODO
# max_input outside of NN
# properly verify inputs
# Properly format output
# higher order polynomial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

# Initialize variables
a, b, c = 5, 0, 0  # polynomial coefficients
training_iterations = 100
lines_to_print = 10
number_to_test = 0.5
max_input = 1


class Net(nn.Module):  # Thanks to some help from my brother Tom
    def __init__(self):
        """Initializing the network"""
        super(Net, self).__init__()
        layer1_neurons = 10
        self.input_layer = nn.Linear(1, layer1_neurons)  # 1 by 10 matrix
        self.hidden_layer = nn.Linear(layer1_neurons, 10)  # 10 by 10 matrix
        self.output_layer = nn.Linear(10, 1)  # 10 by 1 matrix

    def forward(self, vector):
        """Evaluates the net"""
        vector = self.input_layer(vector)
        vector = torch.relu(vector)
        vector = self.hidden_layer(vector)
        vector = torch.relu(vector)
        vector = self.output_layer(vector)
        return vector


def synthetic_target(x):
    """Provides numbers for output"""
    global a, b, c
    return a * x ** 2 + b * x + c


def request_input():
    """Asking for user input to the neural network"""
    global a, b, c, training_iterations, lines_to_print, number_to_test
    print("Create a quadratic function")
    a = float(input("Pick the coefficient of the squared term"))
    b = input("Pick the coefficient of the linear term")
    c = input("Pick the intercept")
    training_iterations = int(input("How many iterations would you like?"))
    lines_to_print = int(input("How many lines would you like to see at the "
                               "beginning and at the end of training?"))
    number_to_test = max_input / 2  # later, let user decide


def verify_input():
    """Veryfing if user input is large enough"""
    global training_iterations
    if training_iterations < 1:
        training_iterations = 100
    else:
        pass


def train_network(iterations=100, lines=20):
    """Training the network"""
    global max_input
    for i in range(training_iterations):

        # Draw random number to seed output
        sample = np.random.rand(1, 1) * max_input

        # Convert to tensor for pytorch usage
        sample = torch.tensor(sample).float()

        actual_output = synthetic_target(sample)
        predicted_output = net(sample)
        loss = loss_function(predicted_output, actual_output)

        # print first and last iterations
        if i < lines_to_print or i > training_iterations - lines_to_print:
            print(str(i), ": Predicted is ", str(predicted_output.item()) +
                  ". Actual value is ", str(
                actual_output.item()) + ". Loss is " + str(loss.item()),
                  sep=" ")  # Prints iteration, predicted value, output, and loss.
        optimizer.zero_grad()
        loss.backward()  # Finds the derivative of the loss in all the parameters
        optimizer.step()  # gradient descent


# Instantiates a Net
net = Net()
loss_function = nn.MSELoss()  # Squared error loss

# Stochastic grad descent
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


def test_network(number_to_test):
    """Using a test value, compares predicted to actual value"""
    global net
    number_to_test = np.random.rand(1, 1) * max_input  # Draw random number to seed output
    number_to_test = torch.tensor(number_to_test).float()
    actual = synthetic_target(number_to_test)
    predicted = net(number_to_test)
    actual_float = synthetic_target(number_to_test).item()
    predicted_float = net(number_to_test).item()

    # print them both, print difference (absolute and percentage)
    print(f"Actual is {round(synthetic_target(number_to_test).item(), 2)}",
          f"Predicted is {round(net(number_to_test).item(), 2)}")
    print(f"Absolute difference is: {round(predicted_float - actual_float, 2)}")
    percent_diff = 100 * abs(predicted_float) / abs(actual_float + predicted_float)
    print(f"Percent difference is: {round(percent_diff, 2)}" + "%")


request_input()
verify_input()

train_network(training_iterations, lines_to_print)
test_network(number_to_test)  # compare network output with polynomial value.

