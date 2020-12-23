import torch
import pdb

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
w = torch.tensor([1.0], requires_grad=True)

def forward(x):
    return x * w

#loss function
def loss(y_pred, y_val):
    return (y_pred-y_val) ** 2

#Before Training
print("Before Training", 4, forward(4).item())

#Training Loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)
        l = loss(y_pred, y_val)
        l.backward()
        print("\tgrad:",x_val,y_val,w.grad.item())
        w.data = w.data - 0.01*w.grad.item()

        w.grad.data.zero_()
    print("Progress:",epoch,l.item())
print("Training Finished",4,forward(4).item())
