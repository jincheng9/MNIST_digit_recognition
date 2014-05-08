% load training set and testing set
clear all;
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

% cliassify the testing set 
train_scale = size(train_set);
test_scale = size(test_set);
test_classify_label = zeros(test_scale(1),1);
tic;
for i=1:test_scale(1)
    test_point = test_set(i,:);
    dist = zeros(train_scale(1),1);
    for j=1:train_scale(1)
        % calculate the distance between test point i and train point j
        train_point = train_set(j, :);
        tmp = test_point - train_point;
		dist(j) = sqrt(sum(tmp.*tmp));
    end
    
    % find the 1-nearest neighbor 
    dist_tmp = sort(dist);
    idx1 = find(dist==dist_tmp(1));
    c1 = train_label(idx1);
    
    % classification 
    test_classify_label(i) = c1;
end
t1 = toc;
% calculate accuracy
num_correct = sum(test_label==test_classify_label);
accuracy = num_correct / test_scale(1);
save -mat 1nn_time.mat t1
save -mat 1nn_accuracy.mat accuracy




