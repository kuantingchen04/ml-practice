from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

## class: {7,9} -> {+1,-1}
def get_y(cls):
    n = cls.shape[0]
    return ((cls==7) * 2 - 1).reshape(n)

train_data = read_csv('data/digits_training_data.csv',header=None,sep=' ').as_matrix()
train_labels = read_csv('data/digits_training_labels.csv',header=None,sep=' ').as_matrix()
train_labels = get_y(train_labels)
num_train,dim_feature = train_data.shape

test_data = read_csv('data/digits_test_data.csv',header=None,sep=' ').as_matrix()
test_labels = read_csv('data/digits_test_labels.csv',header=None,sep=' ').as_matrix()
test_labels = get_y(test_labels)
num_test = test_data.shape[0]


# Cal Best_C using cross-val
#param_grid = {'C':np.linspace(0.1,2.0,num=20)}
param_grid = {'C': np.logspace(-2,4,num=7),'gamma': [1e-2, 1e-3, 1e-4]}
param_grid = {'C': [10],'gamma': [1e-2]} # Best

svc = SVC(random_state=42,kernel='rbf')
#svc.fit(train_data,train_y)
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(train_data, train_labels) # fit data

best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
best_svc = grid_search.best_estimator_

# Eval
train_predict = best_svc.predict(train_data)
train_acc = np.sum(train_predict==train_labels)/train_labels.shape[0]
test_predict = best_svc.predict(test_data)
test_acc = np.sum(test_predict==test_labels)/test_labels.shape[0]
print ("---p4g---") 
print ("best C:",best_C)
print ("best gamma:",best_gamma)
print ("Training Accuracy:",train_acc)
print ("Test Accuracy:",test_acc)
print (best_svc)

# Get misclassified images
mis_idx = np.nonzero(test_predict != test_labels)[0]
plt.figure()
for i,idx in enumerate(mis_idx):
    plt.subplot (3,2,i+1)
    plt.imshow(test_data[idx,:].reshape((28,28)))
    plt.title("label: %s" % (test_labels[idx]+8))
    plt.axis('off')
plt.show()

