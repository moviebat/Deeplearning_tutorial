import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)

    def forward(self, x):
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)
        return y_pred


N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)

model=TwoLayerNet(D_in,H,D_out)
loss_fn=torch.nn.MSELoss(reduction='mean')

learning_rate = 1e-4
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for t in range(500):
    print("epoch is %s" % t)
    y_pred=model(x)
    # print("y_pred is %s" % y_pred)
    loss=loss_fn(y_pred,y)

    if t % 10 == 0:
        print("loss is %s" % loss)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()