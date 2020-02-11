import torch
import encoder
import decoder
import corpus
import os
import time
import jieba
from torch.nn import functional as F
class Teacher():
    def __init__(self,encode_model_path,decode_model_path):
        self.epath = encode_model_path
        self.dpath = decode_model_path
        #设定设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #获取语料库
        self.cp = corpus.Corpus('valid.en-zh.zh.sgm','valid.en-zh.en.sgm','zh_dictionary.pth','en_dictionary.pth')
        
        #初始化超参数  暂定encode与decode网络的嵌入参数和隐层参数一致
        self.embedding_size = 2000
        self.hidden_dim = 500
        self.learning_rate = 0.0002
        self.batch_size = 10
        
        #初始化模型
        self.encode_network = None
        self.decode_network = None
        self.model_init()
        
        #定义优化器 损失函数
        self.encode_optim = torch.optim.SGD(self.encode_network.parameters(), lr = self.learning_rate)
        self.dncode_optim = torch.optim.SGD(self.decode_network.parameters(), lr = self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        #获取文本迭代器
        self.zh = self.cp.sentence_iterator('zh')
        self.en = self.cp.sentence_iterator('en')    
        
        #定义encode初始ht和ct
        self.h0 = torch.zeros(1,1,self.hidden_dim).to(self.device)
        self.c0 = torch.zeros(1,1,self.hidden_dim).to(self.device)
        
        
        
    def model_init(self):
        if self.device == torch.device('cpu'):
            
            if not os.path.exists(self.epath):
                #input_size,embedding_size,hidden_size
                self.encode_network = encoder.Encoder(self.cp.S_len,self.embedding_size,self.hidden_dim)
            if os.path.exists(self.epath):
                self.encode_network = torch.load(self.epath,map_location='cpu')
            
            if not os.path.exists(self.dpath):
                #input_dim,embedding_dim,hidden_dim
                self.decode_network = decoder.Decoder(self.cp.T_len,self.embedding_size,self.hidden_dim)
            if os.path.exists(self.dpath):
                self.decode_network = torch.load(self.dpath,map_location='cpu')
        else:
            if not os.path.exists(self.epath):
                #input_size,embedding_size,hidden_size
                self.encode_network = encoder.Encoder(self.cp.S_len,self.embedding_size,self.hidden_dim).to(self.device)
            if os.path.exists(self.epath):
                self.encode_network = torch.load(self.epath).to(self.device)
            
            if not os.path.exists(self.dpath):
                #input_dim,embedding_dim,hidden_dim
                self.decode_network = decoder.Decoder(self.cp.T_len,self.embedding_size,self.hidden_dim).to(self.device)
            if os.path.exists(self.dpath):
                self.decode_network = torch.load(self.dpath).to(self.device)
                
    def train(self,times):
        
        loss = 0
        batch = 0
        batch_size = self.batch_size
        for i in range(0,times):
            #for j in range(0,self.cp.SentenceNum):
            for j in range(0,500):
                #encoding
                zh_wordlist = jieba.lcut(self.zh[j].childNodes[0].data)
                zh_labels = []
                for word in zh_wordlist:
                    zh_labels.append(self.cp.word2label(word,'zh'))
                zh_labels.append(self.cp.word2label('<end_zh>','zh'))
                zh_inputs = torch.LongTensor(zh_labels).to(self.device)
                zh_out,zh_h_c = self.encode_network(zh_inputs,self.h0,self.c0)
                ht = zh_h_c[0]
                ct = zh_h_c[1]
                #decoding
                #1.预处理输入，添加<start>和<end>
                en_wordlist = (self.en[j].childNodes[0].data).split(' ')
                #print(en_wordlist)
                en_labels = [self.cp.word2label('<start>','en')]                  
                for word in en_wordlist:
                    en_labels.append(self.cp.word2label(word,'en'))
                en_labels.append(self.cp.word2label('<end>','en'))
                en_in_labels = en_labels[0:-1]
                en_out_labels = en_labels[1:]
                
                for lenth in range(0,len(en_in_labels)):
                    en_input = torch.LongTensor([en_in_labels[lenth]]).to(self.device)
                    en_out,en_h_c,attention_out = self.decode_network(en_input,ht,ct,zh_out)
                    #print(attention_out)
                    loss += self.criterion(en_out.squeeze(1),torch.LongTensor([en_out_labels[lenth]]).to(self.device))
                    ht = en_h_c[0]
                    ct = en_h_c[1]
                batch +=1
                if batch>=batch_size:
                    batch = 0
                    loss.backward()
                    self.encode_optim.step()
                    self.dncode_optim.step()
                    loss = 0
                if((j+1)%100 == 0):
                    print('保存模型')
                    torch.save(self.encode_network,self.epath)
                    torch.save(self.decode_network,self.dpath)
                if(j%1 == 0):
                    print('第',i,'轮','第',j,'句','loss:',loss)
        torch.save(self.encode_network,self.epath)
        torch.save(self.decode_network,self.dpath)
    def translate(self,sentence):
        attention_list = []
        wordlist = jieba.lcut(sentence)
        flag = 1
        zh_labels = []
        en_out_labels = []
        en_out_label = self.cp.word2label('<start>','en')
        enend = self.cp.word2label('<end>','en')
        for word in wordlist:
            if self.cp.word2label(word,'zh') == None:
                print(word,'不存在')
                print('翻译失败')
                flag = 0
                return None
            else:
                zh_labels.append(self.cp.word2label(word,'zh'))
                
        if flag:
            zh_labels.append(self.cp.word2label('<end_zh>','zh'))
            zh_inputs = torch.LongTensor(zh_labels).to(self.device)
            zh_out,zh_h_c = self.encode_network(zh_inputs,self.h0,self.c0)
            ht = zh_h_c[0]
            ct = zh_h_c[1]
            lenth = len(zh_labels)
            num = 0
            while(en_out_label!=enend and num<lenth*2):
                num+=1
                en_input = torch.LongTensor([en_out_label]).to(self.device)
                
                en_out,en_h_c,attention_out = self.decode_network(en_input,ht,ct,zh_out)
                attention_list.append(attention_out)
                out = F.softmax(en_out,dim =2)
                out = out.squeeze()
                print(out)
                _,order = torch.sort(out,dim = 0,descending=True)
                print(order)
                en_out_label = order[0].item()
                if en_out_label != enend:
                    en_out_labels.append(en_out_label)
                ht = en_h_c[0]
                ct = en_h_c[1]
            print('翻译句子：',sentence)
            print('翻译为:')
            seg = ''
            for label in en_out_labels:
                seg += self.cp.label2word(label,'en')
                seg+=' '
            print(seg)
            return attention_list
    
        
if __name__ == '__main__':
    teacher = Teacher('encode.pth','decode.pth')
    print(teacher.device)
    teacher.train(30)
    #a = teacher.translate('我是说在干掉他之后')
    #print(a)
    #print(teacher.zh[8].childNodes[0].data)
    #print(teacher.en[8].childNodes[0].data)