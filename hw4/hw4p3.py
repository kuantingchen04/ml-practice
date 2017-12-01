from pandas import read_csv
from numpy.linalg import inv
import numpy as np
from scipy.spatial.distance import pdist, squareform

train_data= read_csv('data/auto_mpg_train.csv',header=None,sep=' ').as_matrix()
num_train = train_data.shape[0]
train_y = train_data[:,0]
train_x = train_data[:,1:]

test_data= read_csv('data/auto_mpg_test.csv',header=None,sep=' ').as_matrix()
num_test = test_data.shape[0]

def get_result(K_inv,kt,train_data,test_data):
    err = 0
    train_y = train_data[:,0]
    for i,data in enumerate(test_data):
        gnd_t = test_data[i,0]
        #kt = train_x.dot(test_data[i,1:])
        pred = (train_y.T).dot(K_inv).dot(kt)
        err = err + pow(gnd_t-pred,2)
    err = np.sqrt(err / test_data.shape[0])
    print (err)

# K_inv: NxN, kt: Nx1
# predict = y' * inv(I+K) + kt


# 0. Origin
K = np.dot(train_x,train_x.T)
K_inv = inv(K + np.identity(num_train))

# Testing
#kt = np.dot(train_x,test_k)
#pred = (train_y.T).dot(K_inv).dot(kt)
err = 0
for i,data in enumerate(test_data):
    gnd_t = test_data[i,0]
    kt = train_x.dot(test_data[i,1:])
    pred = (train_y.T).dot(K_inv).dot(kt)
    err = err + pow(gnd_t-pred,2)
err = np.sqrt(err / num_test)
print (err)

# 1. Poly p=2,4,8
p = 8
K = (np.dot(train_x,train_x.T)+1 ) ** p
K_inv = inv(K + np.identity(num_train))
err = 0
for i,data in enumerate(test_data):
    gnd_t = test_data[i,0]
    kt = (train_x.dot(test_data[i,1:])+1) ** p
    pred = (train_y.T).dot(K_inv).dot(kt)
    err = err + pow(gnd_t-pred,2)
err = np.sqrt(err / num_test)
print (err)

# Gaussian
pairwise_dists = squareform(pdist(train_x, 'euclidean'))
K = np.exp(-pairwise_dists ** 2 / 1 ** 2)
K_inv = inv(K + np.identity(num_train))
err = 0
for i,data in enumerate(test_data):
    gnd_t = test_data[i,0]
    kt = np.exp( -np.sum(train_x-test_data[i,1:],1)**2 / 1**2)
    pred = (train_y.T).dot(K_inv).dot(kt)
    err = err + pow(gnd_t-pred,2)
err = np.sqrt(err / num_test)
print (err)
