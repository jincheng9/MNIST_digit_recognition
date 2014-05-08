% load training set and testing set
clear all;
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');


% training
tic; 
model = svmtrain(train_label, train_set, '-s 0 -t 0');
t1 = toc;
% classification
tic;
[predicted_label, accuracy, decision_values]=svmpredict(test_label, test_set, model);
t2 = toc;
disp(num2str(t1));
disp(num2str(t2));


