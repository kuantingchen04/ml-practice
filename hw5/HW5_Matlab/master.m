% This is the master script for construting and training the neural
% networks. Your work is to implement the gradient descent. 
clear all;
clc;
load data;
rng(42);
%% Construct the network and initialize the parameters
input_dim = 2;
nodes_hidden_1 = 3;
nodes_hidden_2 = 3;
output_dim = 1;

keys = {'W_input_h1', 'b_input_h1', 'W_h1_h2', 'b_h1_h2', 'W_h2_output', 'b_h2_output'};
params = {rand(input_dim, nodes_hidden_1), rand(1, nodes_hidden_1),...
    rand(nodes_hidden_1, nodes_hidden_2), rand(1, nodes_hidden_2),...
    rand(nodes_hidden_2, output_dim), rand(1, output_dim)};
W = containers.Map(keys, params);

%% # Train using Gradient Descent
% training parameters
learning_rate = 0.005;
no_epochs = 2000;

for epoch = 1:no_epochs

%     % Forward propagation
    z_h1 = X * W('W_input_h1') + W('b_input_h1');
    a_h1 = tanh(z_h1);

    z_h2 = a_h1 * W('W_h1_h2') + W('b_h1_h2');
    a_h2 = tanh(z_h2);

    output = a_h2 * W('W_h2_output') + W('b_h2_output');

    probs = sigmoid(output);

    % Backward propagation. YOUR WORK GOES HERE. 
    % Hint: first implement the gradient for each parameter, then apply
    % gradient descent.
    
    %
    % your code......
    %
    
    
    % WORK ENDS HERE.
    % calculate and show the results every 100 iteration.
    loss = cross_entropy_loss(probs, y);    
  
    if mod(epoch, 100) == 0
        fprintf('Loss after epoch %i is %f \n', epoch, loss);
    end
end 
plot_decision_boundary(W, X, y);