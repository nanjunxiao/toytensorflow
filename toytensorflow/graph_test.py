#!/usr/bin/env python
# coding=utf-8

import unittest
from graph import *

class GlobalTest(unittest.TestCase):
    def test_singleton(self):
        graph1 = get_default_graph()
        graph2 = get_default_graph()
        self.assertEqual(graph1.__class__, Graph)
        self.assertEqual(graph2.__class__, Graph)
        self.assertEqual(graph1, graph2)

class GraphTest(unittest.TestCase):
    def test_graph(self):
        default_graph.add_to_graph('add')
        self.assertEqual(default_graph.operations, ['add'])

if __name__ == '__main__':
    unittest.main()
