#!/usr/bin/env python
# coding=utf-8

import numpy as np
import graph
class Op(object):
    def __init__(self, input_nodes=[], name=None):
        self.input_nodes = input_nodes
        for input in input_nodes:
            input.output_nodes.append(self)
        self.output_nodes = []
        self.output_value = None

        self.name = name

        self.graph = graph.get_default_graph()
        self.graph.add_to_graph(self)
        self._isscalar = False 

    def forward(self):
        raise NotImplementedError
    
    def grad(self):
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __div__(self, other):
        return Divide(self, other)

class Add(Op):
    def __init__(self, x, y, name='Add'):
        super(self.__class__, self).__init__([x, y], name=name)

    def forward(self):
        x, y = self.input_nodes
        self.output_value = x.forward() + y.forward()
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = 0
        if len(self.output_nodes) == 0:
            result = np.ones_like(self.output_value)  
        for one in output_nodes_grads:
            result += one
        return [result, result]

class Minus(Op):
    def __init__(self, x, y, name='Minus'):
        super(self.__class__, self).__init__([x, y], name=name)

    def forward(self):
        x, y = self.input_nodes
        self.output_value = x.forward() - y.forward()
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = 0
        if len(self.output_nodes) == 0:
            result = np.ones_like(self.output_value)  
        for one in output_nodes_grads:
            result += one
        return [result, -result]

class Multiply(Op):
    def __init__(self, x, y, name='Multiply'):
        super(self.__class__, self).__init__([x, y], name=name)

    def forward(self):
        x, y = self.input_nodes
        self.output_value = x.forward() * y.forward()
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = 0
        if len(self.output_nodes) == 0:
            result = np.ones_like(self.output_value)  
        for one in output_nodes_grads:
            result += one
        x, y = self.input_nodes
        return [result*y.output_value, result*x.output_value]

class Matmul(Op):
    def __init__(self, x, y, name='Matmul'):
        super(self.__class__, self).__init__([x, y], name=name)

    def forward(self):
        x, y = self.input_nodes
        self.output_value = np.matmul(x.forward(), y.forward() )
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = 0
        if len(self.output_nodes) == 0:
            result = np.ones_like(self.output_value)  
        for one in output_nodes_grads:
            result += one
        x, y = self.input_nodes
        return [np.matmul(result,y.output_value.T), np.matmul(x.output_value.T,result)]

class Square(Op):
    def __init__(self, x, name='Square'):
        super(self.__class__, self).__init__([x], name=name)

    def forward(self):
        x, = self.input_nodes
        self.output_value = np.square(x.forward())
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = 0
        if len(self.output_nodes) == 0:
            result = np.ones_like(self.output_value)  
        for one in output_nodes_grads:
            result += one
        x, = self.input_nodes
        return x.output_value * result * 2

class ReduceMean(Op):
    def __init__(self, x, name='ReduceMean'):
        super(self.__class__, self).__init__([x], name=name)

    def forward(self):
        x, = self.input_nodes
        #TODO
        self.output_value = np.sum(x.forward()) / max(np.shape(x.output_value) )
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = 0
        if len(self.output_nodes) == 0:
            result = np.ones_like(self.output_value)  
        for one in output_nodes_grads:
            result += one
        x, = self.input_nodes
        return result * np.ones_like(x.output_value) / max(np.shape(x.output_value) )

class Variable(Op):
    def __init__(self, value, is_trainable=True, name='Variable'):
        super(self.__class__, self).__init__(name=name)
        #ugly
        self._isscalar = np.isscalar(value)
        self.output_value = value
        if not self._isscalar:
            self.output_value = np.array(value)
        self._is_trainable = is_trainable
        if self._is_trainable:
            self.graph.add_to_trainable_variables(self)

    def forward(self):
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = 0
        if len(self.output_nodes) == 0:
            result = np.ones_like(self.output_value)  
        for one in output_nodes_grads:
            result += one
        if self._isscalar:
            result = np.sum(result) 
        return result

class Constant(Op):
    def __init__(self, value, name='Constant'):
        super(self.__class__, self).__init__(name=name)
        self.output_value = np.array(value)

    def forward(self):
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = np.zeros_like(self.output_value)  
        return result

class Placeholder(Op):
    def __init__(self, dtype=None, shape=None, name='Placeholder'):
        super(self.__class__, self).__init__(name=name)
        self._dtype = dtype 
        self._shape = shape 

    def set_value(self, value):
        self.output_value = np.array(value)

    def forward(self):
        return self.output_value

    def grad(self, output_nodes_grads=None):
        result = 0
        if len(self.output_nodes) == 0:
            result = np.ones_like(self.output_value)  
        for one in output_nodes_grads:
            result += one
        return result

class GlobalVariablesInitializer(Op):
    def __init__(self, name='GlobalVariablesInitializer'):
        super(self.__class__, self).__init__(name=name)

    def forward(self):
        pass

    def grad(self):
        raise NotImplementedError
