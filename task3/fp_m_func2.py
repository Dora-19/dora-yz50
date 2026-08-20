import math

weights = [2.5, 1.5, 3.0] #nöronun kendisine ait bir özellik bu weightler
input_values = [0.5, 1.0, 2.0]
bias = 0.5

target = 0.3

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward_pass(neuron_weights, input_values, bias):
    total = 0
    for i in range(len(neuron_weights)):
        total += neuron_weights[i] * input_values[i]
    return sigmoid(total + bias)

def loss(prediction, target):
    return (prediction - target) ** 2

prediction_for_neuron = forward_pass(weights, input_values, bias)
loss_value = loss(prediction_for_neuron, target)

print("Prediction:", prediction_for_neuron)
print("Loss:", loss_value)