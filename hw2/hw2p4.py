#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import scipy.io as sio

from multiprocessing import Pool

mnist_data = sio.loadmat('Data/mnist_data.mat')
num_sample = 100


def mnist_knn(num_knn):
    # num_knn = 5 
    # Test data for eval
    sampled_idx = np.random.choice(mnist_data['test'].shape[0],num_sample)
    sampled_test_data = mnist_data['test'][sampled_idx,:]
    gt_test = sampled_test_data[:,0]

    # Train data for knn
    gt_train = mnist_data['train'][:,0] 

    # Eval
    acc = 0
    for i in range(num_sample):
        # Infer: find knn
        
        # dist_lst = LA.norm(mnist_data['train'][:,1:] - sampled_test_data[i,1:],axis=1) # Use L2 norm
        dist_lst = np.amax(mnist_data['train'][:,1:] - sampled_test_data[i,1:],axis=1)
        k_idx = dist_lst.argsort()[:num_knn]

        # Eval: compare y_bar & y
        k_labels = gt_train[k_idx]
        k_labels = k_labels.astype(np.int)
        yi_bar = np.argmax(np.bincount( k_labels ))

        if yi_bar == int(gt_test[i]):
            acc += 1
    print("Take k=%s, ACC=%s" % (num_knn,acc/num_sample))


if __name__ == '__main__':

    knn_arr = [1,5,9,13] # k=1,5,9,13
    p = Pool(4)
    p.map(mnist_knn,knn_arr)