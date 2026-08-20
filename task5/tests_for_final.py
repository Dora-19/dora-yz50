import math
import matplotlib.pyplot as plt

weights = [2.5, 1.5, 3.0] #nöronun kendisine ait bir özellik bu weightler
input_values = [0.5, 1.0, 2.0]
bias = 0.5

target = 0.3

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward_pass(neuron_weights, i_values, bias):
    total = 0
    for i in range(len(neuron_weights)):
        total += neuron_weights[i] * i_values[i]
    return sigmoid(total + bias)

def loss(prediction, target):
    return (prediction - target) ** 2

prediction_for_neuron = forward_pass(weights, input_values, bias)

h = 0.0001 # turev hesabı icin cok kucuk degisim

loss_1 = loss(prediction_for_neuron, target)

weights[0] += h
prediction_for_neuron = forward_pass(weights, input_values, bias)

loss_2 = loss(prediction_for_neuron, target)

weights[0] -= h #weighti orjinal haline çevirdik geri

loss_derivative = (loss_2 - loss_1) / h

print("Loss derivative with respect to weight[0]:", loss_derivative)
