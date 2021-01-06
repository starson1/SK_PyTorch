from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() # super : 부모 메소드 상속.
        self.linear = torch.nn.Linear(1, 1) #입력차원, 출력차원을 인수로 받음
        # -> x하나만 입력으로 받고, y하나만 출력으로 뱉기 때문에 (1,1)
 
    def forward(self, x):
        y_pred = self.linear(x) # predict y value
        """
        linear function : 
        """
        return y_pred

model = Model()

criterion = torch.nn.MSELoss(reduction='sum') # MSE : 평균제곱오차
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #업데이트할 parameter, lr : 학습률
"""
   -BATCH : iteration에 사용되는 training set의 묶음.
   -iteration : training 의 반복 횟수. (forward-backward)

BGD : 전체 dataset에 대한 error를 구한뒤 기울기 한번만 계산하여 parameter를 업데이트.
SGD : 추출된 data에 대해서 error를 계산한 뒤, 동일하게 계산하여 parameter를 업데이트

Q1 : 시작지점 어디에 세팅?
Q2 : forward, backward 기능?

참고 : http://sanghyukchun.github.io/74/
"""

# Training loop
for epoch in range(500):
    y_pred = model(x_data) # ???

    loss = criterion(y_pred, y_data) # calculate loss(forward)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    optimizer.zero_grad() # gradient 초기화
    loss.backward() #  
    optimizer.step() #update variable


hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())