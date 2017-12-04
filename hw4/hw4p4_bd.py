from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

train_data = read_csv('data/digits_training_data.csv',header=None,sep=' ').as_matrix()
test_data = read_csv('data/digits_test_data.csv',header=None,sep=' ').as_matrix()
num_train,dim_feature = train_data.shape
num_test = test_data.shape[0]

test_labels = read_csv('data/digits_test_labels.csv',header=None,sep=' ').as_matrix()
train_labels = read_csv('data/digits_training_labels.csv',header=None,sep=' ').as_matrix()

## class: {7,9} -> {+1,-1}
def get_y(origin_labels):
    origin_labels[origin_labels==7] = -1
    origin_labels[origin_labels==9] = 1
    return origin_labels

test_labels = get_y(test_labels)
train_labels = get_y(train_labels)

# Task b: Batch GD
num_epoch = 1000
C = 3
eta = 1e-3

wt = np.zeros(dim_feature)
bt = 0

train_acc_gd = np.empty((num_epoch))
train_acc_gd.fill(np.nan)
for i in range(num_epoch):
    lr = eta / (1 + i * eta)
    
    # Cal grad_w, grad_b
    w_grad, b_grad = wt, 0
    for k,train_x in enumerate(train_data):
        train_y = train_labels[k]
        if train_y*( wt.dot(train_x) + bt ) < 1:
            w_grad = w_grad - C*train_y*train_x
            b_grad = b_grad - C*train_y
    # Update wt, bt
    wtt = wt - lr * w_grad
    btt = bt - lr * b_grad

    # Cal train acc
    pred = train_labels.T * (train_data.dot(wtt) + btt)
    acc = np.sum(pred >= 0) / num_train
    train_acc_gd[i] = acc

    # Check converge
    if LA.norm(wtt-wt)<1e-6:
        break

    wt = wtt
    bt = btt

# --------------------------------------------------------------------------------
# Task d: SGD

wt = np.zeros(dim_feature)
bt = 0

train_acc_sgd = np.empty((num_epoch))
train_acc_sgd.fill(np.nan)
for i in range(num_epoch):
    lr = eta / (1 + i * eta)
    
    # Cal grad_w, grad_b
    permut = np.random.permutation(num_train)
    wtt = wt
    for p_idx in permut:
        w_grad, b_grad = wtt, 0
        train_x = train_data[p_idx,:]
        train_y = train_labels[p_idx]
        if train_y*( wtt.dot(train_x) + bt ) < 1:
            w_grad = w_grad - C*train_y*train_x
            b_grad = b_grad - C*train_y
        # Update wt, bt
        wtt = wtt - lr * w_grad
        btt = btt - lr * b_grad

    # Cal train acc
    pred = train_labels.T * (train_data.dot(wtt) + btt)
    acc = np.sum(pred >= 0) / num_train
    train_acc_sgd[i] = acc

    if LA.norm(wtt-wt)<1e-6:
        break
    wt = wtt
    bt = btt

num_plot = num_epoch
plt.plot()
#plt.plot(train_acc_gd[:numIter],'ro-',markersize=2)
#plt.plot(train_acc_sgd[:numIter],'bo-',markersize=2)
plt.plot(train_acc_gd[:num_plot],'r')
plt.plot(train_acc_sgd[:num_plot],'b')
plt.xlabel('Iteration (Epoch)')
plt.ylabel('Train Acc')
plt.axis([0,num_plot,0.5,1])
plt.legend(['BGD','SGD'])
plt.show()
