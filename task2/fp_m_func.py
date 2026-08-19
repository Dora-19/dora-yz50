import math

weights = [[2.5, 1.5, 3.0], [2.5, 1.5, 3.0], [2.5, 1.5, 3.0]] #nöronun kendisine ait bir özellik bu weightler
input_values = [0.5, 1.0, 2.0]
biases = [0.5, 1.0, 1.5]

outputs = []

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward_pass(neuron_weights, input_values, bias):
    total = 0
    for i in range(len(neuron_weights)):
        total += neuron_weights[i] * input_values[i]
    return sigmoid(total + bias)

for i in range(len(weights)):
    output = forward_pass(weights[i], input_values, biases[i])
    outputs.append(output)

print(outputs)

