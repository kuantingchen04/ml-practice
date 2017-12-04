from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

train_data = read_csv('data/synthetic.csv',header=None).as_matrix()
num_train = train_data.shape[0]
train_x = train_data[:,0].reshape((num_train,1))
train_y = train_data[:,1].reshape((num_train,1))

train_x = np.hstack((train_x,np.ones((num_train,1))))

# C:N*N (diag)
# w_opt = inv(x.T*C*x)*x.T*C*Y
# w -> r -> C -> w ...

# Robust LS
wt = np.zeros((2,1))
for iter_i in range(100):
    c_ii = np.zeros(num_train)
    for i in range(num_train):
        r_i = train_y[i] - (wt.T).dot(train_x[i])
        c_ii[i] = (1 + r_i**2) ** (-1/2)

    C = np.diag(c_ii)
    wtt = inv((train_x.T).dot(C).dot(train_x)).dot((train_x.T).dot(C).dot(train_y))
    
    if np.mean((wt-wtt)**2) < 1e-6:
        break
    wt = wtt
wt_robust = wt

# OLS
wt = np.zeros((2,1))
for iter_i in range(100):
    c_ii = np.zeros(num_train)
    for i in range(num_train):
        #r_i = train_y[i] - (wt.T).dot(train_x[i])
        r_i = 2
        c_ii[i] = (1 + r_i**2) ** (-1/2)

    #C = np.diag(c_ii)
    C = np.eye(num_train) # OLS
    wtt = inv((train_x.T).dot(C).dot(train_x)).dot((train_x.T).dot(C).dot(train_y))
    
    if np.mean((wt-wtt)**2) < 1e-6:
        break
    wt = wtt
wt_ols = wt


plt.figure()
a_x = np.array([-100,1])
b_x = np.array([100,1])
w_gt = np.array([10,5])
plt.plot([a_x[0],b_x[0]],[a_x.dot(wt_robust),b_x.dot(wt_robust)],'g-')
plt.plot([a_x[0],b_x[0]],[a_x.dot(wt_ols),b_x.dot(wt_ols)],'b-')
plt.plot([a_x[0],b_x[0]],[a_x.dot(w_gt),b_x.dot(w_gt)],'r-')
plt.scatter(train_x[:,0],train_y,s=2)
plt.legend(['Robust LS','OLS','True Line'])
#plt.plot([150,360],[260,50],'k-')
plt.axis([-.5,2,-20,40])
print ("Robust LS:\n",wt_robust)
print ("OLS:\n", wt_ols)
plt.show()


