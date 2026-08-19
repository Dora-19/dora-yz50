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

i = -100
plot_pred = []
plot_loss = []
plot_w = []

while i <=100:
    weights[0] = i
    plot_w.append(i)

    prediction_for_neuron = forward_pass(weights, input_values, bias)
    plot_pred.append(prediction_for_neuron)

    loss_value = loss(prediction_for_neuron, target)
    plot_loss.append(loss_value)
    i += 0.1

plt.plot(plot_w, plot_loss)
plt.xlabel("weights")
plt.ylabel("loss")
plt.show()


