import torch
import xml.dom.minidom
import jieba
import os

'''
类:Corpus
主要功能：维护处理好的字典与提供一切语料相关的信息
'''

class Corpus():
    def __init__(self,source,target,spath,tpath):
        self.source = source
        self.target = target
        self.spath = spath
        self.tpath = tpath
        self.S_set = set()
        self.T_set = set()
        self.S_dictionary = []
        self.T_dictionary = []
        self.S_len = 0
        self.T_len = 0
        self.zh_init()
        self.en_init()
        self.special_word_append('<start>','en')
        self.special_word_append('<end>','en')
        self.special_word_append('<end_zh>','zh')
        self.SentenceNum = self.__sentence_num()
    def word2label(self,word,s_or_t):
        if s_or_t == 'zh':
            for item in self.S_dictionary:
                if item[0] == word:
                    return item[1]
        if s_or_t == 'en':
            for item in self.T_dictionary:
                if item[0] == word:
                    return item[1]
        return None
            
    def label2word(self,label,s_or_t):
        if s_or_t == 'zh':
            for item in self.S_dictionary:
                if item[1] == label:
                    return item[0]
        if s_or_t == 'en':
            for item in self.T_dictionary:
                if item[1] == label:
                    return item[0]
        return None
    
    def zh_init(self):
        if not os.path.exists(self.spath):
            DOMTree = xml.dom.minidom.parse(self.source)
            root = DOMTree.documentElement
            segs = root.getElementsByTagName('seg')
            for seg in segs:
                sentence = seg.childNodes[0].data
                print(sentence)
                wordlist = jieba.lcut(sentence)
                for word in wordlist:
                    self.S_set.add(word)
            num = 0
            for word in self.S_set:
                self.S_dictionary.append([word,num])
                num+=1
            self.S_len = len(self.S_dictionary)
            torch.save(self.S_dictionary,self.spath)
        else:
            self.S_dictionary = torch.load(self.spath)
            self.S_len = len(self.S_dictionary)
    
    def en_init(self):
        if not os.path.exists(self.tpath):
            DOMTree = xml.dom.minidom.parse(self.target)
            root = DOMTree.documentElement
            segs = root.getElementsByTagName('seg')
            for seg in segs:
                sentence = seg.childNodes[0].data
                wordlist = sentence.split(' ')
                for word in wordlist:
                    self.T_set.add(word)
            num = 0
            for word in self.T_set:
                self.T_dictionary.append([word,num])
                num+=1
            self.T_len = len(self.T_dictionary)
            torch.save(self.T_dictionary,self.tpath)
        else:
            self.T_dictionary = torch.load(self.tpath)
            self.T_len = len(self.T_dictionary)
    
    def special_word_append(self,word,s_or_t):
        if s_or_t == 'zh':
            if self.word2label(word,'zh') == None:
                self.S_dictionary.append([word,self.S_len])
                self.S_len +=1
            else:
                print(word,'已经存在')
        if s_or_t == 'en':
            if self.word2label(word,'en') == None:
                self.T_dictionary.append([word,self.T_len])
                self.T_len +=1
            else:
                print(word,'已经存在')
                
    def sentence_iterator(self,s_or_t):
        if s_or_t == 'zh':
            DOMTree = xml.dom.minidom.parse(self.source)
            root = DOMTree.documentElement
            segs = root.getElementsByTagName('seg')
            return segs
        
        if s_or_t == 'en':
            DOMTree = xml.dom.minidom.parse(self.target)
            root = DOMTree.documentElement
            segs = root.getElementsByTagName('seg')
            return segs
    
    def __sentence_num(self):
        b = self.sentence_iterator('zh')
        num = 0
        for item in b:
            num+=1
        return num
    
if __name__ == "__main__":
    corpus = Corpus('valid.en-zh.zh.sgm','valid.en-zh.en.sgm','zh_dictionary.pth','en_dictionary.pth')
    #print(corpus.word2label('boss','en'))
    bbb = corpus.sentence_iterator('zh')
    for i in bbb:
        print (i.childNodes[0].data)
    for i in bbb:
        print('sb')
    #for i in bbb:
       # print(i.childNodes[0].data)
    #print(bbb[0].childNodes[0].data)
    print(corpus.SentenceNum)
    



    
