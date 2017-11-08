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
l_prev = 1000;
test_cost = [];
train_cost = [];

% Run iter
cnt=0;
for step=1:100
    
    h=0; grad=zeros(num_dim,1); hessian=zeros(num_dim,num_dim);
    l_train=0; train_acc=0;
    for i=1:num_train
        x = train_data(i,:);
        y = train_labels(i);
        h = 1 / (1+exp( -dot(wk,x) ));
        grad = grad + (h - y)*x';
        hessian = hessian + h*(1-h)*(x')*x;
        l_train = l_train - (y*log(h)+(1-y)*log(1-h));
        train_acc = train_acc + double((h>=0.5 && y==1)||(h<=0.5 && y==0));
    end
%     train_acc/num_train
    train_cost = [ train_cost l_train ];
    
    % Test loss (cost)
    l_test = 0;
    test_acc = 0;
    for ii=1:num_test
        x = test_data(ii,:);
        y = test_labels(ii);
        h = 1 / (1+exp( -dot(wk,x) ));
        l_test = l_test - (y*log(h)+(1-y)*log(1-h));
        test_acc = test_acc + double((h>=0.5 && y==1)||(h<=0.5 && y==0));
    end
%     test_acc/num_test
    test_cost = [ test_cost l_test ];    
    
    % Coverage check (loss diff)
    abs(l_train-l_prev)
    if abs(l_train-l_prev) < 1e-8 
        break
    end

    wk = wk - inv(hessian)*grad; cnt=cnt+1; % update weight
    
    l_prev = l_train;
end

% Draw
figure();
plot(train_cost,'-bo','LineWidth',2)
hold on;
plot(test_cost,'-ro','LineWidth',2)
hold off;
legend('Train loss','Test loss')
xlabel('Iteration')
ylabel('Train/Test Loss')
