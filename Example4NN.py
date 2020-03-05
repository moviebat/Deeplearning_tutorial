import torch
from torch.autograd import Variable

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)

model=torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
		torch.nn.ReLU(),
		torch.nn.Linear(H,D_out))
loss_fn=torch.nn.MSELoss(reduction='mean')

learning_rate = 1e-4
for t in range(500):
    print("epoch is %s" % t)
    y_pred=model(x)
    #print("y_pred is %s" % y_pred)
    loss=loss_fn(y_pred,y)

    if t % 10 == 0:
        print("loss is %s" % loss)

    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.data-=learning_rate*param.grad.data