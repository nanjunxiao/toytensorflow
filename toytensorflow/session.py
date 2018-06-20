#!/usr/bin/env python
# coding=utf-8

import operations 
class Session(object):
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        pass

    def run(self, op, feed_dict=None):
        if feed_dict != None:
            for k,v in feed_dict.items():
                k.set_value(v)
        result = op.forward()
        return result
