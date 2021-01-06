from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() # super : 부모 메소드 상속.
        self.linear = torch.nn.Linear(1, 1) #입력차원, 출력차원을 인수로 받음

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

criterion = torch.nn.MSELoss(reduction='sum')
print(list(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # lr : 학습률

# Training loop
for epoch in range(500):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())