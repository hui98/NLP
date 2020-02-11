import torch

b = 'ss aa bb'
c = b.split(' ')
print(c)
'''
encodeing_hid = torch.randn([6,1,3])

decodeing_inp = torch.randn([1,1,4])
w1 = torch.nn.Linear(4,3)
out1 = w1(decodeing_inp)

hid = encodeing_hid.squeeze(1)
out1 = out1.squeeze(1).transpose(0,1)
out2 = torch.mm(hid,out1)

out3 = F.softmax(out2,dim = 0)

print(hid)
print(encodeing_hid)
print(out1)
print(out2)
print(out3.size())
model = TransModel(5,2,3,5)
print(model.rnn_encode.weight)
'''
'''
aaa = torch.nn.LSTM(5,3)

bbb = torch.randn([2,1,5])
hi =torch.randn([1,1,3])
ci = torch.randn([1,1,3])
print(hi)
out,h_c = aaa(bbb,(hi,ci))
print(bbb)

print(out)

print(h_c[1])
'''