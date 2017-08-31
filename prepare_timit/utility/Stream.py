# coding = utf-8

import re
import sys
import shlex
import subprocess

class Stream :
    def __init__(self) :
        self.kind = ''
        self.space = []
        self.cost = {}
        self.cost['cor'] = 0
        self.cost['sub'] = 1
        self.cost['del'] = 1
        self.cost['ins'] = 1

