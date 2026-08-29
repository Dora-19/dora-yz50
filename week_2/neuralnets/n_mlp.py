import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import random

class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0

        self._backward = lambda: None #YENI

        self._prev = set(_children) #sonucun hangi değerlerden geldiğini tutar
        self._op = _op #sonucun hangi işlem sonucu oluştuğunu tutar
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})" #print(a) yazdığında ne görüneceğini belirler. Tanımlamazsan <__main__.Value object at 0x104a2f> gibi bi şey görürsün
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = _backward

        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        
        return output
    
    def __pow__(self, other):   #YENI
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data**(other-1)) * output.grad
        output._backward = _backward

        return output
    
    def __radd__(self, other): #YENI
        return self + other 

    def __rmul__(self, other):  #YENI
        return self * other
    
    def __truediv__(self, other):  #YENI
        return self * other**-1
    
    def __neg__(self):  #YENI
        return self * -1
    
    def __sub__(self, other):   #YENI
        return self + (-other)
    
    def exp(self):
        x = self.data
        output = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += output.data * output.grad
        output._backward = _backward

        return output
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        output = Value(t, (self, ), "tanh")

        def _backward():
           self.grad += (1 - t**2) * output.grad
        output._backward = _backward

        return output
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

        

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



class Neuron:
  
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
  
  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]


class Layer:
  
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
  
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  

# x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1]) # 3 inputs, 2 hidden layers of width 4, and 1 output
# draw_dot(n(x)).render('week_2/neuralnets/mlp_neuron.gv', view=True)

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
# [(yout - ygt)**2 for yout, ygt in zip(ypred, ys)] #MSE loss

for k in range(20):

    #forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt)**2 for yout, ygt in zip(ypred, ys)])

    #backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    #uypdate
    for p in n.parameters():
        p.data += -0.05 * p.grad  # gradient descent

    print(k, loss.data)
    
# draw_dot(loss).render('week_2/neuralnets/mlp_loss.gv', view=True)