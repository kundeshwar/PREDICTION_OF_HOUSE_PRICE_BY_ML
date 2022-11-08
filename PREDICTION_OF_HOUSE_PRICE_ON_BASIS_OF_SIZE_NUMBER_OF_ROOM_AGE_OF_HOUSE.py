#multiple varible  linear regression
import numpy as np
import matplotlib.pyplot as plt
import math, copy

x_train = np.array([[2104, 5, 1, 45],[1416, 3, 2, 40], [852, 2, 1, 35]])#size, number of bathroom, number of floors, age of home
y_train = np.array([460, 232, 178])#price in dollers
print(x_train)
print(y_train)
b_init = 785.181136
w_init = np.array([ 0.39133535, 18.7576741, -53.36032453, -26.42131])#number of len(x_train)+len(y_train)
print(f"w_init shape: {w_init}, b_init type :{type(b_init)}")
def predict_single_loop(x, w, b):
    n = x.shape[0]
    p =0
    for i in range(n):
        p_i = x[i]*w[i]
        p += p_i
    p = p+b
    return p
x_vec = x_train[0, :]
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f_wb,"==f_wb")
#single prdection vector
def predict(x, w, b):
    p = np.dot(x, w)+b
    return p

f_wb = predict(x_vec, w_init, b_init)
print(f_wb,"==f_wb")

#computing cost 
def comput_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost = cost + ((f_wb_i-y[i])**2)
    cost = cost/(2*m)
    return cost

cost = comput_cost(x_train, y_train, w_init, b_init)
print(cost)

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

tmp_dj_dw, temp_dj_db = computer_gradient(x_train, y_train, w_init, b_init)
print("kund")
print(tmp_dj_dw, temp_dj_db)

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

intial_w = np.zeros_like(w_init)
intial_b = 0.
iteration = 1000
alpha = 5.0e-7
w_final, b_final, j_hist = Gradient_Descent(x_train, y_train, intial_w, intial_b, comput_cost, computer_gradient, alpha, iteration)
print(f"b_final value : {b_final:0.2f}, {w_final}")
m,_ = x_train.shape
for i in range(m):
    b_1 = np.dot(x_train[i], w_final)
    print(f"this is prediction: {b_1 + b_final:0.2f}", y_train[i])

x_train_1 = np.array([1000, 3, 1, 38])


b_1 = np.dot(x_train_1, w_final)
print(f"this is prediction: {b_1 + b_final:0.2f}")