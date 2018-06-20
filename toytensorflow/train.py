#!/usr/bin/env python
# coding=utf-8

import operations
from Queue import Queue
import numpy as np
class GradientDescentOptimizer(object):
    def __init__(self, learning_rate):
        self._learing_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self._learing_rate
        class MinimizationOp(operations.Op):
            def forward(self):
                #add forwad
                loss.forward()
                grad_map = compute_grad(loss)
                for var in self.graph.trainable_variables:
                    grad = grad_map[var]
                    var.output_value -= learning_rate * grad

        return MinimizationOp()

def compute_grad(target_op):
    grad_table = {}

    #BFS for redundant computation 
    queue = Queue()
    queue.put(target_op)

    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()
        output_nodes_grads = []
        for output_node in node.output_nodes:
            if len(output_node.input_nodes) > 1:
                idx = output_node.input_nodes.index(node)
                output_nodes_grads.append(grad_table[output_node][idx])
            else:
                output_nodes_grads.append(grad_table[output_node])
        grad_table[node] = node.grad(output_nodes_grads)
        for input_node in node.input_nodes:
            if input_node not in visited:
                queue.put(input_node)
                visited.add(input_node)

    return grad_table
