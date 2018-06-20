#!/usr/bin/env python
# coding=utf-8

class Graph(object):
    def __init__(self):
        self.trainable_variables = []
        self.operations = []

    def add_to_graph(self, op):
        self.operations.append(op)

    def add_to_trainable_variables(self, op):
        self.trainable_variables.append(op)

default_graph = Graph() 
def get_default_graph():
    global default_graph
    if default_graph == None:
        default_graph = Graph()
    return default_graph
