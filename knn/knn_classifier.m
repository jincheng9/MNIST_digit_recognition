% load and divide the training set 
clear all;
data = load('train.mat');
data = data.train;
data_scale = size(data);
train_set = data(1:data_scale(1)*2/3, :);
test_set = data(data_scale(1)*2/3+1:data_scale(1), :);

% classify the testing set 
train_scale = size(train_set);
test_scale = size(test_set);
test_label = zeros(test_scale(1),1);
for i=1:test_scale(1)
    test_point = test_set(i,2:test_scale(2));
    dist = zeros(train_scale(1),1);
    label = zeros(train_scale(1), 1);
    for j=1:train_scale(1)
        % calculate the distance between test point i and train point j
        train_point = train_set(j, 2:train_scale(2));
        dis = 0.0;
        for k=1:test_scale(2)-1
            dis  = dis + (test_point(k)-train_point(k))*(test_point(k)-train_point(k));
        end
        dist(j) = sqrt(dis);
        label(j) = train_set(j,1);        
    end
    dist = sort(dist);
    idx1 = find(dist==dist(1));
    idx2 = find(dist==dist(2));
    idx3 = find(dist==dist(3));
    c1 = label(idx1);
    c2 = label(idx2);
    c3 = label(idx3);
    if(c1==c2||c1==c3) 
        test_label(i) = c1;
    elseif (c2==c3)
        test_label(i) = c2;
    else test_label(i) = c3;
    end
end

% calculate accuracy
num_correct = sum(test_set(:,1)==test_label);
accuracy = num_correct / test_scale(1);





