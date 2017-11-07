#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt

## Naive Bayes Implementation

word_idx = np.array(range(1,1449))
def cal_error(train_file,test_file):
    ## 3a: Read training data
    word_idx = np.array(range(1,1449))
    train_dicts = [{ x:1 for x in word_idx},{ x:1 for x in word_idx} ] # init both cnt with 1 -> Laplace smoothing
    trainReader = csv.reader(open(train_file, newline=''), delimiter=' ', quotechar='|')
    for row in trainReader:
        if int(row[0])==1: 
            for wc in row[2:]:
                key,value = wc.split(":")
                key,value = int(key),int(value)
                train_dicts[0][key] = train_dicts[0][key] + value
        elif int(row[0])==-1:
            for wc in row[2:]:
                key,value = wc.split(":")
                key,value = int(key),int(value)
                train_dicts[1][key] = train_dicts[1][key] + value
    sum_spam = sum(train_dicts[0].values())
    sum_nospam = sum(train_dicts[1].values())

    # Test, Evaluation
    error_cnt = 0
    #testReader = csv.reader(open('data/spam_classification/SPARSE.TEST', newline=''), delimiter=' ', quotechar='|')
    testReader = csv.reader(open(test_file, newline=''), delimiter=' ', quotechar='|')
    for row in testReader:
        w_given_spam = 0
        w_given_nospam = 0
        gt = int(row[0])
        for wc in row[2:]:
            key,value = wc.split(":")
            key,value = int(key),int(value)
            w_given_spam = w_given_spam + value*np.log(train_dicts[0][key]/sum_spam)
            w_given_nospam = w_given_nospam + value*np.log(train_dicts[1][key]/sum_nospam)

        if (w_given_spam>=w_given_nospam and gt==-1) or (w_given_spam<=w_given_nospam and gt==1):
            error_cnt = error_cnt + 1
    error = error_cnt / testReader.line_num *100
    #print ("Error Rate: %s %%" % error)
    return error,sum_spam,sum_nospam,train_dicts

print ("---hw2-p3a---")
train_lst = ['.50','.100','.200','.400','.800','.1400','']
train_n_lst = [50,100,200,400,800,1400,1448]
error_lst = []
train_path = 'data/spam_classification/SPARSE.TRAIN'
test_path = 'data/spam_classification/SPARSE.TEST'
for i,n in enumerate(train_lst):
    error,_,_,_ = cal_error(train_path + n,test_path)
    error_lst.append(error)
print (error_lst)
plt.figure(1)
plt.plot(train_n_lst,error_lst,marker='o',linestyle='-',color='r')
plt.ylabel("Error Rate")
plt.xlabel("Training Data Size")
plt.draw()
plt.pause(0.5)

## Q3b: Finding 5-most indicative tokens
error,sum_spam,sum_nospam,train_dicts = cal_error(train_path,test_path)
score_list = np.zeros(len(word_idx))
for i,w_idx in enumerate(word_idx):
    score_list[i] = np.log( (train_dicts[0][w_idx]/sum_spam) / (train_dicts[1][w_idx]/sum_nospam) )
    s_idx = score_list.argsort()[-5:]

print ("---hw2-p3b---")
for line in open('data/spam_classification/TOKENS_LIST'):
    if (int(line.split()[0])-1) in s_idx:
        top_5_result = line.split()[1]
        print (top_5_result)

plt.figure(2)
plt.plot(word_idx, score_list, 'o', markersize=2)
plt.plot(word_idx[s_idx], score_list[s_idx], 'ro', markersize=5)

plt.axis([0, 1449, -10, 10])
plt.xlabel("Word Index")
plt.ylabel("Indicative Score")
plt.show()

