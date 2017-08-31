#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

if __name__ == '__main__':
    data = {}
    for line in open(sys.argv[1], 'r') :
        line = line.decode('utf-8').rstrip('\n')
        key = line[0:line.find(' ')]
        data[key] = line
    for key in sys.stdin :
        key = key.decode('utf-8').rstrip('\n')
        print(data[key].encode('utf-8'))

