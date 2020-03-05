import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader


# A DataLoader wraps a Dataset and provides minibatching,
# shuffling, multithreading, for you
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

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

loader=DataLoader(TensorDataset(x,y),batch_size=8)

model=TwoLayerNet(D_in,H,D_out)
loss_fn=torch.nn.MSELoss(reduction='mean')

learning_rate = 1e-4
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for t in range(100):
    print("epoch is %s" % t)
    for x_batch,y_batch in loader:
        x_var,y_var=Variable(x),Variable(y)

        y_pred=model(x_var)
        # print("y_pred is %s" % y_pred)
        loss = loss_fn(y_pred,y)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    if t % 10 == 0:
        print("loss is %s" % loss)