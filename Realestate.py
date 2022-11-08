
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
data = pd.read_csv("realestate.csv")
#a = data.loc[1:3]#it is used to finding rows 
#print(a)
a = data.columns
print(a)
#x_train, y_train = data
y_train = np.array(data["Y house price of unit area"])
print(y_train)
print(data.shape)
d = []
for i in range(0, 414):
    #print(i)
    b = [data["X1 transaction date"].loc[i], data['X2 house age'].loc[i], data['X3 distance to the nearest MRT station'].loc[i], data['X4 number of convenience stores'].loc[i], data['X5 latitude'].loc[i], data['X6 longitude'].loc[i]]
    #print(b)
    d.append(b)

x_train = np.array(d)
print(x_train)
#i = 0
#np.append(x_train,[data["X1 transaction date"].loc[i], data['X2 house age'].loc[i], data['X3 distance to the nearest MRT station'].loc[i], data['X4 number of convenience stores'].loc[i], data['X5 latitude'].loc[i], data['X6 longitude'].loc[i]] )
#print(x_train)
#print(data)
#i = 0
#print([data["X1 transaction date"].loc[i], data['X2 house age'].loc[i], data['X3 distance to the nearest MRT station'].loc[i], data['X4 number of convenience stores'].loc[i], data['X5 latitude'].loc[i], data['X6 longitude'].loc[i]])
w_init = np.array([0.39133, 18.75376, 5.986, -53.36032, -26.4213, -0.596])
b_init = 785.18
#-----------------------1
#computing cost 
def comput_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost = cost + ((f_wb_i-y[i])**2)
    cost = cost/(2*m)
    return cost
#computing and display cost using our pre-chosen optimal parameters .
cost = comput_cost(x_train, y_train, w_init, b_init)
print(cost)
#----------------------2
#Gradient Descent with multiple Varible 
#derivate function
def computer_gradient(x, y, w, b):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(x[i], w)+b)-y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j]+(err*x[i, j])
        dj_db = dj_db+err
    dj_db = dj_db/m
    dj_dw = dj_dw/m
    return dj_dw, dj_db
#compute and display gradient
tmp_dj_dw, tmp_dj_db = computer_gradient(x_train, y_train, w_init, b_init)
print(f"tmp_dj_dw:-{tmp_dj_dw}, tmp_dj_db:-{tmp_dj_db}")
#--------------------------3
import math, copy
#Gradient Descent with multiple varible
def Gradient_Descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - (alpha*dj_dw)
        b = b- (alpha*dj_db)
        if i < 100000:
            j_history.append( cost_function(x, y, w, b))
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4d}: cost{j_history[-1]:8.2f}")
    return w, b, j_history

alpha = 1e-7
initial_w= np.zeros_like(w_init)
initial_b = 0.
iteration = 1000
w_final, b_final, j_history = Gradient_Descent(x_train, y_train, initial_w, initial_b, comput_cost, computer_gradient, alpha, iteration)
print(w_final, b_final)
#predition
print(np.dot(x_train[1], w_final)+b_final)
print(y_train[1])








