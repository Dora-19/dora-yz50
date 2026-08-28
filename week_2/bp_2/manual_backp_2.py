import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children) #sonucun hangi değerlerden geldiğini tutar
        self._op = _op #sonucun hangi işlem sonucu oluştuğunu tutar
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})" #print(a) yazdığında ne görüneceğini belirler. Tanımlamazsan <__main__.Value object at 0x104a2f> gibi bi şey görürsün
    
    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), "+")
        return output
    
    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), "*")
        return output
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")
        return out
        

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid() #tanhı görmek için

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# bias of the neuron
b = Value(6.8813735870195432, label='b')

# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

#tanh implement
o = n.tanh(); o.label = 'o'

#gradları bulalım
o.grad = 1.0

# o = tanh(n)
# do/dn = 1 - tanh(n)^2 = 1 - o^2
print(1 - o.data**2) #do/dn
n.grad = 0.5

#toplama node u distribute ediyodu gradi, yine 0.5 olacak.
x1w1x2w2.grad = 0.5
b.grad = 0.5

#aynı mevzu
x1w1.grad = 0.5
x2w2.grad = 0.5

#son aşama
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

dot = draw_dot(o)
dot.render('graph', view=True)