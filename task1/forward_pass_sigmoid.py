import math
weight = 2.5
input_value = 0.5
bias = 0

output = weight * input_value + bias

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

forward_output = sigmoid(output)

print(forward_output)


#Relu Version

# import math
# weight = 2.5
# input_value = 0.5
# bias = 0

# output = weight * input_value + bias

# def relu(x):
#     if x <= 0:
#         return 0
#     else:
#         return x

# forward_output = relu(output)

# print(forward_output)