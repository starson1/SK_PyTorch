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

w_list=[]
mse_list=[]

for w in np.arange(0.0,4.1,0.1):
    print("w=",round(w,2))
    l_sum = 0

    for x_val, y_val in zip(x_data,y_data):
        y_pred_val = forward(x_val)
        l = loss(x_val,y_val)
        l_sum += l
        print("\t",round(x_val,2),round(y_val,2),round(y_pred_val,2),round(l,2))
    MSE = round((l_sum / 3),2)
    print("MSE=",MSE)
    w_list.append(w)
    mse_list.append(MSE)
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()