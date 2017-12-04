from pandas import read_csv
from numpy.linalg import inv
import numpy as np
from scipy.spatial.distance import pdist, squareform

train_data = read_csv(
    'data/auto_mpg_train.csv',
    header=None,
    sep=' ').as_matrix()
test_data = read_csv(
    'data/auto_mpg_test.csv',
    header=None,
    sep=' ').as_matrix()
test_data = test_data - np.mean(train_data, 0)
train_data = train_data - np.mean(train_data, 0)  # Demean

num_train = train_data.shape[0]
num_test = test_data.shape[0]
train_y = train_data[:, 0]
train_x = train_data[:, 1:]

# Learning
# K_inv: NxN, kt: Nx1

# Testing
# kt = np.dot(train_x,test_k)
# pred = (train_y.T).dot(K_inv).dot(kt)

# 0. Origin
K = np.dot(train_x, train_x.T)
K_inv = inv(K + np.identity(num_train))

err = 0
for i, data in enumerate(train_data):
    gnd_t = data[0]
    kt = train_x.dot(data[1:])
    pred = (train_y.T).dot(K_inv).dot(kt)
    err = err + (gnd_t - pred) ** 2
err = np.sqrt(err / num_train)
print("Ordinary:\t%s" % err)

# 1. Poly p=2,4,8
p = 2
K = (np.dot(train_x, train_x.T) + 1) ** p
K_inv = inv(K + np.identity(num_train))
err = 0
for i, data in enumerate(test_data):
    gnd_t = data[0]
    kt = (train_x.dot(data[1:]) + 1) ** p
    pred = (train_y.T).dot(K_inv).dot(kt)
    err = err + (gnd_t - pred) ** 2
err = np.sqrt(err / num_test)
print("Poly using p=%s: %s" % (p, err))

## 2. Poly p=2,4,8
p = 4
K = (np.dot(train_x, train_x.T) + 1) ** p
K_inv = inv(K + np.identity(num_train))
err = 0
for i, data in enumerate(test_data):
    gnd_t = data[0]
    kt = (train_x.dot(data[1:]) + 1) ** p
    pred = (train_y.T).dot(K_inv).dot(kt)
    err = err + (gnd_t - pred) ** 2
err = np.sqrt(err / num_test)
print("Poly using p=%s: %s" % (p, err))

# 3. Poly p=2,4,8
p = 8
K = (np.dot(train_x, train_x.T) + 1) ** p
K_inv = inv(K + np.identity(num_train))
err = 0
for i, data in enumerate(test_data):
    gnd_t = data[0]
    kt = (train_x.dot(data[1:]) + 1) ** p
    pred = (train_y.T).dot(K_inv).dot(kt)
    err = err + (gnd_t - pred) ** 2
err = np.sqrt(err / num_test)
print("Poly using p=%s: %s" % (p, err))

#
# Gaussian
pairwise_dists_mat = squareform(pdist(train_x, 'euclidean'))
K = np.exp(-pairwise_dists_mat ** 2 / (2 * 1 ** 2))
K_inv = inv(K + np.identity(num_train))
err = 0
for i, data in enumerate(test_data):
    gnd_t = data[0]
    kt = np.exp(- np.sum((train_x - data[1:]) ** 2, 1) / (2 * (1 ** 2)))
    pred = (train_y.T).dot(K_inv).dot(kt)
    err = err + (gnd_t - pred)**2
err = np.sqrt(err / num_train)
print("Gaussian:\t%s" % err)
