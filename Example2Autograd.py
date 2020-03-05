import torch
from torch.autograd import Variable

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(N,D_in),requires_grad=False)
y=Variable(torch.randn(N,D_out),requires_grad=False)
w1=Variable(torch.randn(D_in,H),requires_grad=True)
w2=Variable(torch.randn(H,D_out),requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    print("epoch is %s" % t)
    y_pred=x.mm(w1).clamp(min=0).mm(w2)
    # print("y_pred is %s" % y_pred)
    loss=(y_pred-y).pow(2).sum()

    if t!=0:
        w1.grad.data.zero_()
        w2.grad.data.zero_()

    if t % 10 == 0:
        print("loss is %s" % loss)
    loss.backward()

    w1.data-=learning_rate*w1.grad.data
    w2.data-=learning_rate*w2.grad.data