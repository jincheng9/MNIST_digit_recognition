<h1>Digit Recognizer for MNIST Data Set</h1>
In this project, I try different classification methods to recognize the digits in the MNIST data set.

<h2>MNIST Data Set</h2>
There are 60000 training samples and 10000 testing samples in the MNIST data set. 
Detailed description about this data set could be found via http://yann.lecun.com/exdb/mnist/index.html. 
Notice that the MNIST data set has been normalized to [0 1]. 
<h2>Features</h2>
Each digit image is 28x28 pixel. 
In the data set, each sample is represented by a normalized 28x28=784 dimensional vector. I directly use this vector as the feature vector of each sample. 
<h2>Classification Methods</h2>
<ul>
<li>
KNN: <br>
accuracy 97.05%, no training time, testing time &#77min
</li>
<li>
Linear kernel SVM: <br>
accuracy 93.98%, training time &#8776 6.658min, testing_time &#8776 2.76min
</li>
<li>
Polynomial kernel SVM with degree 2: <br>
accuracy 98.08%, training time &#8776 3.9min, testing_time &#8776 2.4min
</li>
<li>
Radial basis kernel SVM with default gamma: <br>
accuracy 94.46%, training time &#8776 10.43min, testing time &#8776 4.84min
</li>
<li>
Artificial Neural Network with 1 hidden layer (784-300-10): <br>
1 iteration: accuracy 87.17% <br>
20 iterations: accuracy 96.19% <br>
100 iterations: accuracy 97.94% <br>
</li>
<li>
Convolutional Neural Network: <br>
1 epoch: accuracy 88.17%, training time + testing time &#8776 90s <br>
100 epochs: accuracy 98.85%, training time + testing time &#8776 2hr 30min
</li>
</ul>

<h2>Directories included in the toolbox</h2>
load_data/ - data set and functions to load the data <br>
knn/ - 3-nearest neighborhood classifier <br>
svm/ - A library of linear, polynomial and RBF classifier based on libsvm <br>
ann/ - A library of fully connected feedforward neural network and convolutional neural network (CNN is based on DeepLearnToolbox) <br>

<h2> Setup</h2>
1. Download <br>
2. addpath(genpath('MNIST_digit_recognition'));

<h2> Example </h2>
```Matlab
% load training set and testing set
clear all;
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

% parameter setting
alpha = 0.1; % learning rate
beta = 0.01; % scaling factor for sigmoid function
train_size = size(train_set);
N = train_size(1); % number of training samples
D = train_size(2); % dimension of feature vector
n_hidden = 300; % number of hidden layer units
K = 10; % number of output layer units
% initialize all weights between -1 and 1
W1 = 2*rand(1+D, n_hidden)-1; % weight matrix from input layer to hidden layer
W2 = 2*rand(1+n_hidden, K)-1; % weight matrix from hidden layer to ouput layer
max_iter = 100; % number of iterations
Y = eye(K); % output vector 


% training 
for i=1:max_iter
	disp([num2str(i), ' iteration']);
    for j=1:N
        % propagate the input forward through the network
        input_x = [1; train_set(j, :)'];
        hidden_output = [1;sigmf(W1'*input_x, [beta 0])];
        output = sigmf(W2'*hidden_output, [beta 0]);
        % propagate the error backward through the network
        % compute the error of output unit c
        delta_c = (output-Y(:,train_label(j)+1)).*output.*(1-output);
        % compute the error of hidden unit h
        delta_h = (W2*delta_c).*(hidden_output).*(1-hidden_output);
        delta_h = delta_h(2:end);
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
    hidden_output = [1; sigmf(W1'*input_x, [beta 0])];
    output = sigmf(W2'*hidden_output, [beta 0]);
    [max_unit, max_idx] = max(output);
    if(max_idx == test_label(i)+1)
        num_correct = num_correct + 1;
    end
end
% computing accuracy
accuracy = num_correct/test_size(1);
```

<h2> References </h2>
<ul>
<li>http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset </li>
<li>https://github.com/cjlin1/libsvm</li>
<li>https://github.com/rasmusbergpalm/DeepLearnToolbox</li>
</ul>

<h2> Remark </h2>
If you are testing CNN using Octave, please pay attention to the Octave version because there is a bug in the previous versions.
<br>
error: Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions.

