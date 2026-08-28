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

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b
e.label = 'e'
d = e + c
d.label = 'd'
f = Value(-2.0, label='f')
L = d * f
L.label = 'L'

#bu değerlerden bazıları weightler bazıları da inputlar olacaklar ve derrivativelerini bularak aslında her birinin Loss functiona nasıl etki ettiğini buluyoruz.
#Inputlar genelde fixed ama weightleri oynayarak Loss func ı minimize edebiliriz. Amaç bu.
#bu yüzden grafikte bir değerın yanındaki grad aslında L nin o değere göre türevi aslında.

L.grad = 1.0 # kendisine göre türevi

# for d's grad value, we need dL/dd = ??
# def of der ====> f(x + h) - f(x)) / h

f.grad = 4.0
d.grad = -2.0

#numerical derrivate cok ufak h ekleyip yaptığımız türev hesabı
#dL/dc lazim simdi c.grad için. Bunun için de Chain rule yapicaz (dL/dd) * (dd/dc) = dL/dc

#dplus nodes chain ruleda 1.0 ile çarpmak oluyo derrivatives just got routed to both e and c
#bunlar local derrivativeler, rest of thje grapjhi bilmiyo embed edildiği sadece lokal influence biliyo.

c.grad = -2.0
e.grad = -2.0

# a ve b yi bulucaz şimdi.
#dL/da = dL/de * de/da = -2.0 * -3.0 = 6.0
#dL/db = dL/de * de/db = -2.0 * 2.0 = -4.0

a.grad = (-2.0 * -3.0)
b.grad = (-2.0 * 2.0)

dot = draw_dot(L)
dot.render('graph', view=True)

#L i arttırmak için grad yönünde gidiyoruz
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * e.grad
f.data += 0.01 * f.grad

e = a * b
d = e + c
L = d * f

print(L.data)  #/Documents/GitHub/dora-yz50/week_2/manual_backp_1.py     -7.286496
