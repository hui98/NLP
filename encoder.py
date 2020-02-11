import torch
from torch.nn import functional as F

class Encoder(torch.nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size):
        super(Encoder,self).__init__()
        self.embedding = torch.nn.Embedding(input_size,embedding_size)
        self.rnn_encode = torch.nn.LSTM(embedding_size,hidden_size)
    
    def forward(self,inputs,h0,c0):
        #print(inputs)
        embed = self.embedding(inputs)
        #print(embed)
        embed = embed.unsqueeze(1)
        out,h_c = self.rnn_encode(embed,(h0,c0))
        return out,h_c

if __name__ == "__main__":
    model = Encoder(20,10,5)

    h0 = torch.zeros(1,1,5)
    c0 = torch.zeros(1,1,5)
    inputs = torch.LongTensor([1,2,3])

    out,h_c = model(inputs,h0,c0)
    print(h_c[0])

