import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

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
# o = n.tanh(); o.label = 'o'   #BUNU DEGISTIRIYORUZ YENI YAZDIGIMIZ KOMPONENTLERLE

e = (2*n).exp()
o = (e - 1) / (e + 1)

# o.grad = 1.0
# o._backward()
# n._backward()

# b._backward
# x1w1x2w2._backward()

# x2w2._backward()
# x1w1._backward()

# o.grad = 1.0

# #topological order
# topo = []
# visited = set()
# def build_topo(v):
#   if v not in visited:
#     visited.add(v)
#     for child in v._prev:
#       build_topo(child)
#     topo.append(v)
# build_topo(o)

# for node in reversed(topo):
#   node._backward()

# print(topo)

o.backward()

dot = draw_dot(o)
dot.render('graph', view=True)