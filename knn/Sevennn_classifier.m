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
    
    % find the 5-nearest neighbor 
    dist_tmp = sort(dist);
    num = zeros(10, 1);
    for k=1:7
       idx = find(dist==dist_tmp(k));
       num(train_label(idx)+1) = num(train_label(idx)+1)+1;
    end
    
    % classification
    maxIdx = 0;
    maxNum = -1;
    for k=1:10
      if(num(k)>maxNum)
         maxIdx = k;
         maxNum = num(k);
      end
    end 
    test_classify_label(i) = maxIdx-1;
end
t1 = toc;
% calculate accuracy
num_correct = sum(test_label==test_classify_label);
accuracy = num_correct / test_scale(1);
save -mat 7nn_time.mat t1
save -mat 7nn_accuracy.mat accuracy




