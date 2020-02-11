import torch
from torch.nn import functional as F
class Decoder(torch.nn.Module):
    def __init__(self,input_dim,embedding_dim,hidden_dim):
        super(Decoder,self).__init__()
        self.embedding = torch.nn.Embedding(input_dim,embedding_dim)
        self.linear1 = torch.nn.Linear(hidden_dim,hidden_dim)
        self.linear2 = torch.nn.Linear(embedding_dim+hidden_dim,embedding_dim)
        self.rnn =  torch.nn.LSTM(embedding_dim,hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim,input_dim)
    
    def forward(self,input,hidden_t,c_t,encode_out):
        embed = self.embedding(input)
        w_dhid = self.linear1(hidden_t)
        w_dhid = w_dhid.squeeze(1)
        ehid = encode_out.squeeze(1)
        ehid_trans = ehid.transpose(0,1)
        ehid_w_dhid = torch.mm(w_dhid,ehid_trans)
        ehid_w_dhid = F.softmax(ehid_w_dhid,dim = 1)
        #print(ehid_w_dhid)
        attention_out = torch.mm(ehid_w_dhid,ehid)
        attention_out = attention_out.unsqueeze(1)
        #print(attention_out)
        embed = embed.unsqueeze(1)
        combination = torch.cat((attention_out,embed),2)
        rnn_input = self.linear2(combination)
        rnn_input = F.relu(rnn_input)
        out,h_c = self.rnn(rnn_input,(hidden_t,c_t))
        out = self.linear3(out)
        return out,h_c,ehid_w_dhid
        
        
if __name__ == "__main__":
    model = Decoder(10,8,5)
    a = [1,2,3]
    input = torch.LongTensor([a[1]])
    hidden_t = torch.zeros([1,1,5])
    c_t = torch.zeros([1,1,5])
    encode_out = torch.randn([4,1,5])
    out,h_c,attention_out = model(input,hidden_t,c_t,encode_out)
    print(out)
    print(h_c)
    print(attention_out)