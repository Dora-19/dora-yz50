import math

weights = [2.5, 1.5, 3.0] #nöronun kendisine ait bir özellik bu weightler
input_values = [0.5, 1.0, 2.0]
bias = 0.5
total = 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# weights[0] * input_values[0] + biases[0] ----> bu 1. işlem olacak ve böyle böyle gidecek. for kullanalım.

for i in range(len(weights)):
    total += weights[i] * input_values[i]

forward_output = sigmoid(total + bias)

print(forward_output)

#burdaki input values girdiler, bütün işlemler nöron içi hesaplamalar, forward output da nöronun çıktısı. nöron dediğin şey bu ara işlemler yani aslında.
#bias eklerken başta kafamn karıştı,i her nöron için bir bias olması gerekirken ben üç bias ekledim ve üç farklı bias değeriyle toplamış oldum her turda. hatalı.
#nöronun agırlık sayısı = nöronun girdi sayısı, seçme hakkın yok.