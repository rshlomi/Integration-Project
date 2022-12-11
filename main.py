# Ron Shlomi
# Artificial Neural Network
__author__ = "Ron Shlomi"

# TODO
# max_input outside of [0,1]
# Properly format output
# Plot actual and predicted polynomial for 0-1 range
# make code more efficient

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt

# Code to satisfy requirements
print("Welcome to my CS project!" * 5, end="!!!!!!!\n")  # Prints the string five times
a = 3
b = 7
print(b // a)


# Initialize variables
training_iterations = 100
max_input = 1
losses = list()

# define class and functions
class Net(nn.Module):
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

def synthetic_target(x, coefficients):
    """Provides numbers for output"""
    degree = len(coefficients)
    result = 0
    for i in range(degree):
        result += coefficients[i] * x ** i
    return result

def request_input():
    """Asking for user input to the neural network"""
    coefficients = []
    degree = -1
    while degree < 0:
        degree = int(input("Enter the degree of the polynomial: "))

    # Request the coefficients from the user
    for i in range(degree + 1):
        coefficient =\
          float(input("Enter the coefficient for the x^" + str(i) + " term: "))
        coefficients.append(coefficient)
    training_iterations = 1
    while not isinstance(training_iterations, int) or training_iterations < 10:
        try:
            training_iterations = int(input("How many iterations would you like? "))
            if training_iterations < 100:
                print("Please enter a number larger than 100")
                training_iterations = 1
            elif training_iterations > 10000:
                print("Please enter a number smaller than 10000")
                training_iterations = 1
            else:
                pass
        except:
            print("Please enter an integer")
    print(f"The network will be trained to emulate polynomials in the (0, {max_input}) domain.")
    number_to_test = 2
    print_every = -9
    while print_every < 0:
        try:
            print_every =\
                int(input("Please enter how often you want to print iterations "))
        except:
            print("Please enter an integer")
    # To force an and into my program
    while not(not number_to_test < 0 and not number_to_test > max_input):
            try:
                number_to_test =\
                float(input("Please enter a number to test between 0 and 1 "))
            except:
                print("Please enter a numbe between 0 and 1")
    return coefficients, training_iterations, number_to_test, print_every


def train_network(max_input, iterations=100, lines=20):
    """Training the network"""
    for i in range(training_iterations):

        # Draw random number to seed output
        sample = np.random.rand(1, 1) * max_input

        # Convert to tensor for pytorch usage
        sample = torch.tensor(sample).float()

        actual_output = synthetic_target(sample, coefficients)
        predicted_output = net(sample)
        loss = loss_function(predicted_output, actual_output)
        losses.append(loss.item())

        # print_every iteration
        if i % print_every == 0:
            print(str(i), ": Predicted is ", str(predicted_output.item()) +
                ". Actual value is ", str(
                actual_output.item()) + ". Loss is " + str(loss.item()),
                sep=" ")  # Prints iteration, predicted value, output, and loss.
        else:
            pass
        optimizer.zero_grad()
        loss.backward()  # Finds the derivative of the loss in all the parameters
        optimizer.step()  # gradient descent


def plot_loss(losses):
    """Plots the loss over time"""
    plt.plot(losses)
    plt.title("Performance improvement")
    plt.xlabel("Iteration")
    plt.ylabel('Loss')
    plt.show()

def test_network(number_to_test, net):
    """Using user's test value, compares predicted to actual value"""
    number_to_test = np.array([number_to_test])
    number_to_test = torch.tensor(number_to_test).float()
    predicted = net(number_to_test)
    actual_float = synthetic_target(number_to_test, coefficients).item()
    predicted_float = net(number_to_test).item()

    txt0 = "Actual is: {actual_num:.2f}"
    txt1 = "Predicted is: {predicted_num:.2f}"
    txt2 = "Absolute difference is: {abs_diff:.2f}"
    txt3 = "Percent difference is: {cent_diff:.2f} %"

    # print them both, print difference (absolute and percentage)
    print(txt0.format(actual_num=synthetic_target(number_to_test,\
                                                  coefficients).item()),
          txt1.format(predicted_num=net(number_to_test).item()))
    print(txt2.format(abs_diff=(predicted_float - actual_float)))
    print(txt3.format(cent_diff=(100 * abs(predicted_float - actual_float)\
                                  / actual_float)))

# Instantiates a Net
net = Net()

# Squared error loss
loss_function = nn.MSELoss()

# Stochastic grad descent
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

coefficients, training_iterations, number_to_test, print_every = request_input()
print("Input verified")
print("-" * 79)
train_network(max_input, training_iterations, print_every)
# Pretty print the polynomial using string formatting and list comprehension
polynomial_string = " + ".join([str(coefficients[i]) + "x^" + str(i) \
                    for i in range(len(coefficients)) if coefficients[i] != 0])
print("The polynomial is:", polynomial_string)
test_network(number_to_test, net)  # compare network output with polynomial value.
plot_loss(losses)
