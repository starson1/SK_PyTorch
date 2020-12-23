import numpy as np
import matplotlib.pyplot as plt

w= 1.0 #a random guess: random value
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward(x):
    return x*w

#loss function
def loss(x,y):
    y_pred = forward(x)
    return pow((y_pred-y),2)

#gradient
def gradient(x,y):
    return (2*x*(x*w-y))

w_list=[]
mse_list=[]

#training loop
for epoch in range(100):
    for x_val,y_val in zip(x_data,y_data):
        grad = gradient(x_val,y_val)
        w -= 0.01*grad #w configuration
        print("\tgrad:",x_val,y_val,grad)
        l = loss(x_val,y_val)
           
    print("progress :",epoch, "w=",w,"loss=",round(l,20))
print("Training Finished : 4 hours ",forward(4))


"""
1 : 2x(x^2w_2 + xw_1 + b - y)
2 : 2x^2(x^2w_2 + xw_1 + b - y)
"""
