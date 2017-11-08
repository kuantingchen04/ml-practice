from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from pandas import read_csv

## Read data (768 -> last 268:testing)
data_df = read_csv('data/diabetes_scale.csv',header=None)
data = data_df.as_matrix()
test_data = data[500:,1:]
train_data = data[:500,1:]
test_label = data[500:,0]
train_label = data[:500,0]

# Cal Best_C using cross-val
param_grid = {'C':np.linspace(0.1,2.0,num=20)}

svc = LinearSVC(random_state=42,loss='hinge')
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(train_data, train_label) # fit data

best_C = grid_search.best_params_['C']
best_svc = grid_search.best_estimator_

# Eval
# p5d1
train_predict = best_svc.predict(train_data)
train_acc = np.sum(train_predict==train_label)/train_label.shape[0]
test_predict = best_svc.predict(test_data)
test_acc = np.sum(test_predict==test_label)/test_label.shape[0]
print ("---p5d1---") 
print ("best C:",best_C)
print ("Training Accuracy:",train_acc)
print ("Test Accuracy:",test_acc)
print (best_svc.coef_)

# p5d2
svc = LinearSVC(C=1e6,random_state=42,loss='hinge')
svc.fit(train_data,train_label)

test_predict = svc.predict(test_data)
test_acc = np.sum(test_predict==test_label)/test_label.shape[0]
train_predict = best_svc.predict(train_data)
train_acc = np.sum(train_predict==train_label)/train_label.shape[0]
print ("---p5d2---") 
print ("C:",1e6)
print ("Training Accuracy:",train_acc)
print ("Test Accuracy:",test_acc)
print (svc.coef_)

