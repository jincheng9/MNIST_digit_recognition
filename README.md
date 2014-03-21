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
accuracy 97.04%, no training time, testing time &#8776 3hr
</li>
<li>
Linear kernel SVM: <br>
accuracy 93.98%, training time &#8776 10min, testing_time &#8776 3min
</li>
<li>
Polynomial kernel SVM with degree 2: <br>
accuracy 91.35%, training time &#8776 20min, testing_time &#8776 8min
</li>
<li>
Radial basis kernel SVM with default gamma: <br>
accuracy 94.46%, training time + testing time &#8776 20min
</li>
<li>
Artificial Neural Network with 1 hidden layer (784-300-10): <br>
100 iterations: accuracy 97.51%, training time + testing time &#8776 50min <br>
300 iterations: accuracy 97.56%, training time + testing time &#8776 2hr 30min
</li>
<li>
Convolutional Neural Network: <br>
1 epoch: accuracy 88.17%, training time + testing time &#8776 90s <br>
100 epochs: accuracy 98.85%, training time + testing time &#8776 2hr 30min
</li>
</ul>

<h2>Usage</h2>
Since I use stanford's matlab function "loadMNISTImages" and "loadMNISTLabels" to load the data, you need to download theese two functions first. The link is as follows:
http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset <br>

If you want to test a specific classification method of mine, you need to put loadMNISTImages.m, loadMNISTLabels.m, the data set and my code in the same directory. 
And then you just need to type the file name to run my code in MATLAB. I add detailed comments for my code, which should be useful if you want to modify my code. 


