from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim

x_data = tensor([[2.1], [4.2], [3.1], [3.3]])
y_data = tensor([[0.], [1.], [0.], [1.]])

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear = nn.Linear(1,1)
    
    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = Model()

criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(),lr = 0.01)
#training loop
for epoch in range(1000):
    y_pred = model(x_data)

    loss = criterion(y_pred,y_data)
    print(f'Epoch{epoch+1}/1000 | Loss: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#prediction loop
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]])) # 여기서 x 값이 의미하는 거 : ???
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[3.1]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')