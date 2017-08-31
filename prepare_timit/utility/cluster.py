# coding = utf-8

from Accuracy import is_chinese
import re
import sys

if __name__ == '__main__':
    cluster = {}
    cluster['ch'] = {}
    cluster['en'] = {}
    for line in sys.stdin :
        line = line.decode('utf-8')
        word = line[0:line.find(' ')-1]
        if is_chinese(word) :
            for char in word :
                cluster['ch'][char] = 1
        else :
            cluster['en'][word] = 1
    print('<English>')
    for token in cluster['en'] :
        print(token.encode('utf-8'))
    print('<Mandarin>')
    for token in cluster['ch'] :
        print(token.encode('utf-8'))

