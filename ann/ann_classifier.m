% load training set and testing set
clear all;
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

% parameter setting
alpha = 0.1; % learning rate
train_size = size(train_set);
N = train_size(1); % number of training samples
D = train_size(2); % dimension of feature vector
n_hidden = 300; % number of hidden layer units
K = 10; % number of output layer units
% initialize all weights between -1 and 1
W1 = 2*rand(1+D, n_hidden)-1; % weight matrix from input layer to hidden layer
W2 = 2*rand(n_hidden, K)-1; % weight matrix from hidden layer to ouput layer
max_iter = 100; % number of iterations
Y = eye(K); % output vector 

% training 
for i=1:max_iter
    for j=1:N
        % propagate the input forward through the network
        input_x = [1; train_set(j, :)'];
        hidden_output = sigmf(W1'*input_x, [1 0]);
        output = sigmf(W2'*hidden_output, [1 0]);
        % propagate the error backward through the network
        % compute the error of output unit c
        delta_c = (output-Y(:,train_label(j)+1)).*output.*(1-output);
        % compute the error of hidden unit h
        delta_h = (W2*delta_c).*(hidden_output).*(1-hidden_output);
        % update weight matrix
        W1 = W1 - alpha*(input_x*delta_h');
        W2 = W2 - alpha*(hidden_output*delta_c');
    end
end

% testing 
test_size = size(test_set);
num_correct = 0;
for i=1:test_size(1)
    input_x = [1; test_set(i,:)'];
    hidden_output = sigmf(W1'*input_x, [1 0]);
    output = sigmf(W2'*hidden_output, [1 0]);
    [max_idx, max] = max(output);
    if(max_idx == test_label(i)+1)
        num_correct = num_correct + 1;
    end
end
% accuracy
accuracy = num_correct/test_size(1);


