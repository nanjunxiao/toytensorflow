#!/usr/bin/env python
# coding=utf-8

import unittest
from operation import *

class OpTest(unittest.TestCase):
    def test_super(self):
        graph1 = get_default_graph()
        graph2 = get_default_graph()
        self.assertEqual(graph1.__class__, Graph)
        self.assertEqual(graph2.__class__, Graph)
        self.assertEqual(graph1, graph2)

if __name__ == '__main__':
    unittest.main()
