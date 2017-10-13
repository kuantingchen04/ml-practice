#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import scipy.io as sio
from multiprocessing import Pool
import matplotlib.pyplot as plt

# Q4
mnist_data = sio.loadmat('data/mnist_data.mat')
num_sample = 1000

def l1(diff):
    return LA.norm(diff,axis=1)
def l2(diff):
    return np.sum(np.abs(diff),axis=1)

def mnist_knn(num_knn,dist_f):
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
        dist_lst = dist_f(mnist_data['train'][:,1:] - sampled_test_data[i,1:])
        k_idx = dist_lst.argsort()[:num_knn]

        # Eval: compare y_bar & y
        k_labels = gt_train[k_idx]
        k_labels = k_labels.astype(np.int)
        yi_bar = np.argmax(np.bincount( k_labels ))

        if yi_bar == int(gt_test[i]):
            acc += 1

    print("Take k=%s, Test Accuracy=%s" % (num_knn, acc/num_sample))
    return acc/num_sample

if __name__ == '__main__':

    knn_arr = [1,5,9,13] # k=1,5,9,13
    l1_arr,l2_arr = [l1]*4,[l2]*4
    p = Pool(4)
    print("---hw2-p4a---") 
    l1_r = p.starmap(mnist_knn,zip(knn_arr,l1_arr))
    l2_r = p.starmap(mnist_knn,zip(knn_arr,l2_arr))
    print(l1_r)
    print(l2_r)

    plt.plot(knn_arr, l1_r,marker='x',linestyle='-',color='b',label="L1")
    plt.plot(knn_arr, l2_r,marker='o',linestyle='-',color='r',label="L2")
    plt.ylim([0.5,1])
    plt.xlabel("kNN")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.show()




