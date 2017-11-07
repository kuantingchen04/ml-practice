addpath data/classification
load('train_features.dat')
load('train_labels.dat')
load('test_features.dat')
load('test_labels.dat')

% Concat
num_train = size(train_features,1);
num_test = size(test_features,1);
train_data = [ ones(num_train,1) train_features ];
test_data = [ ones(num_test,1) test_features ];
num_dim = size(train_data,2);

% Init
wk = [0 0 0]';
l = 0;

test_err = [];
train_err = [];

% Run iter
for step=1:10
    
    h=0; grad=zeros(num_dim,1); hessian=zeros(num_dim,num_dim);
    ll=0;
    for i=1:num_train
        x = train_data(i,:);
        y = train_labels(i);
        h = 1 / (1+exp( -dot(wk,x) ));
        grad = grad + (h - y)*x';
        hessian = hessian + (h*(1-h))* x'*x;
        
        ll = ll - (y*log(h)+(1-y)*log(1-h));
    end
    wkk = wk - hessian'*grad; % update weight
    
    % Train & Test error
    train_err = [ train_err ll ];
    
    l_test = 0;
    
    for i=1:num_test
        x = test_data(i,:);
        y = test_labels(i);
        h = 1 / (1+exp( -dot(wkk,x) ));
        l_test = l_test - (y*log(h)+(1-y)*log(1-h));
    end
    test_err = [ test_err l_test ];
    
    % Compare loss
    if norm(ll-l) < 1e-8 
%     if norm(grad) < 1e-8
        norm(ll-l);
        break
    end
    
    wk = wkk;
    l = ll;
    
end

% Draw
wk
step
train_err
test_err
