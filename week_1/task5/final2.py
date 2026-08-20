import math
import matplotlib.pyplot as plt

weights = [-25, 1.5, 3.0] #nöronun kendisine ait bir özellik bu weightler
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

h = 0.0001 # turev hesabı icin cok kucuk degisim

step = 0
learning_rate = 10 # ilk başta 0.1 di ve kayda değer düşüşler olmadı, 100 yaptim cukurdan atladı
loss_ds = []

for step in range(2000):

    predict_1 = forward_pass(weights, input_values, bias)
    loss_1 = loss(predict_1, target)

    weights[0] += h

    predict_2 = forward_pass(weights, input_values, bias)
    loss_2 = loss(predict_2, target)

    weights[0] -= h #weighti orjinal haline çevirdik geri

    loss_derivative = (loss_2 - loss_1) / h

    weights[0] = weights[0] - learning_rate * loss_derivative

forward_pass_result = forward_pass(weights, input_values, bias)
print("final loss:", loss(forward_pass_result, target))
print("Final weight:", weights[0])
print("final_der",loss_derivative)