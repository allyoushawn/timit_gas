# coding = utf-8

class Vocabulary:
    def __init__(self) :
        self.data = []
    def load(self, fn) :
        for line in open(fn, 'r'):
            line = line.decode('utf-8')
            idx = line.find(' ')
            if idx == -1 : 
                token = line.rstrip('\n')
            else :
                token = line[0:idx]
            length = len(token)
            while len(self.data) <= length :
                self.data.append({})
            if(token in self.data[length]) :
                self.data[length][token] = self.data[length][token] + 1
            else :
                self.data[length][token] = 1
    def save(self, fn) :
         f = open(fn, 'w')
         for arr in self.data :
             for key in arr.keys() :
                 f.write(key.encode('utf-8'))
                 f.write('\n')
         f.close()
    def oov(self, token) :
        if len(token) >= len(self.data) :
            return True
        if token in self.data[len(token)] : 
            return False
        else :
            return True
    def segment(self, string) :
        if len(string) == 0 :
            return string
        head = -1
        size = len(self.data)
        for size in reversed(xrange(len(self.data))) :
            if len(string) < size :
                continue
            for key in self.data[size].keys() :
                head = string.find(key)
                if head >= 0 : 
                    break
            if head >= 0 :
                break
                    
        if head == -1 : # string is oov
            return string
        if size == len(string) : # exact match
            return string
        l = ''
        m = ''
        r = ''
        idx = 0
        for char in string :
            if idx < head :
                l += char
            elif idx < head+size :
                m += char
            else :
                r += char
            idx = idx + 1
        l = self.segment(l)
        m = self.segment(m)
        r = self.segment(r)
        result = l
        if len(l) > 0 :
            result += ' '
        result += m
        if len(r) > 0 :
            result += ' '
        result += r
        return result

def is_chinese(word):
    for char in word :
        if char < u'\u4e00' or char > u'\u9fa5':
            return False
    return True
# 
#def is_number(uchar):
#        if uchar >= u'\u0030' and uchar<=u'\u0039':
#                return True
#        else:
#                return False
# 
#def is_alphabet(uchar):
#        if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
#                return True
#        else:
#                return False
# 
#def is_other(uchar):
#        if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
#                return True
#        else:
#                return False
