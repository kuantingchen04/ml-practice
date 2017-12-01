from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

train_data = read_csv('data/digits_training_data.csv',header=None,sep=' ').as_matrix()
train_labels = read_csv('data/digits_training_labels.csv',header=None,sep=' ').as_matrix()
num_train,dim_feature = train_data.shape

test_data = read_csv('data/digits_test_data.csv',header=None,sep=' ').as_matrix()
test_labels = read_csv('data/digits_test_labels.csv',header=None,sep=' ').as_matrix()
num_test = test_data.shape[0]

## class: {7,9} -> {+1,-1}
def get_y(cls):
    return (cls[0]==7) * 2 - 1

# Task b: Batch GD
num_iter = 20
C = 3
eta = 1e-3

wt = np.zeros(dim_feature)
bt = 0

train_acc_gd = []
for i in range(num_iter):
    lr = eta / (1 + i * eta)
    
    # Cal grad_w, grad_b
    w_grad, b_grad = wt, 0
    for k,train_x in enumerate(train_data):
        train_y = get_y(train_labels[k])
        if train_y*( wt.dot(train_x) + bt ) < 1:
            w_grad = w_grad - C*train_y*train_x
            b_grad = b_grad - C*train_y
    # Update wt, bt
    wtt = wt - lr * w_grad
    btt = bt - lr * b_grad
    # Cal train acc
    cnt = 0
    for k,train_x in enumerate(train_data):
        train_y = get_y(train_labels[k])
        if train_y*( wt.dot(train_x) + bt )>0: # Inference
           cnt = cnt + 1
    train_acc_gd.append(cnt/num_train)
    print ("Iter: %s \t Train acc: %s" % (i,cnt / num_train))
    wt = wtt
    bt = btt

# --------------------------------------------------------------------------------
# Task d: SGD
num_iter = 20
C = 3
eta = 1e-3

wt = np.zeros(dim_feature)
bt = 0

train_acc_sgd = []
for i in range(num_iter):
    lr = eta / (1 + i * eta)
    
    # Cal grad_w, grad_b
    w_grad, b_grad = wt, 0
    permut = np.random.permutation(num_train)
    #for k,train_x in enumerate(train_data):
    for k in permut:
        train_x = train_data[k,:]
        train_y = get_y(train_labels[k])
        if train_y*( wt.dot(train_x) + bt ) < 1:
            w_grad = w_grad - C*train_y*train_x
            b_grad = b_grad - C*train_y
        # Update wt, bt
        wt = wt - lr * w_grad
        bt = bt - lr * b_grad

    # Cal train acc
    cnt = 0
    for k,train_x in enumerate(train_data):
        train_y = get_y(train_labels[k])
        if train_y*( wt.dot(train_x) + bt )>0: # Inference
           cnt = cnt + 1
    train_acc_sgd.append(cnt/num_train)
    print ("Iter: %s \t Train acc: %s" % (i,cnt / num_train))


plt.plot()
plt.plot(train_acc_gd,'ro-',markersize=5)
plt.plot(train_acc_sgd,'bo-',markersize=5)
plt.xlabel('Iteration')
plt.ylabel('Train Acc')
plt.axis([0,num_iter,0,1])
plt.legend(['GD','SGD'])
plt.show()
