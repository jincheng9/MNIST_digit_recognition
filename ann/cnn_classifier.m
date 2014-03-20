% load training set and testing set
clear all;
train_set = loadMNISTImages('train-images.idx3-ubyte')'; % 60000 x 784
train_label = loadMNISTLabels('train-labels.idx1-ubyte'); % 60000 x 1
test_set = loadMNISTImages('t10k-images.idx3-ubyte')'; % 10000 x 784
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte'); % 10000 x 1

% trasform the data format 
N = 10; % N is the number of output neurons
Y = eye(N);
train_size = size(train_set);
test_size = size(test_set);
train_y = zeros(train_size(1), N); % train_label to train_y 60000 x 10
test_y = zeros(test_size(1), N); % test_label to test_y 10000 x 10
for i=1:train_size(1)
    train_y(i,:) = Y(train_label(i)+1,:);
end
for i=1:test_size(1)
    test_y(i,:) = Y(test_label(i)+1,:);
end

% use the code of DeepLearnToolbox
train_x = double(reshape(train_set',28,28,60000));
test_x = double(reshape(test_set',28,28,10000));
train_y = double(train_y');
test_y = double(test_y');

% parameter setting
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 100;

cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);

assert(er<0.12, 'Too big error');
