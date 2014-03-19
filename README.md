<h1>Digit Recognizer for MNIST Data Set</h1>
In this project, I try different classification methods to recognize the digits in the MNIST data set.

<h2>MNIST Data Set</h2>
There are 60000 training samples and 10000 testing samples in the MNIST data set. 
Detailed description about this data set could be found via http://yann.lecun.com/exdb/mnist/index.html. 
<h2>Features</h2>

<h2>Classification Methods</h2>
<ul>
<li>
KNN: accuracy 97.04%, running time &#8776 3hr
</li>
<li>
Linear kernel SVM: accuracy 93.98%, training time &#8776 10min, testing_time &#8776 3min
</li>
<li>
Polynomial kernel SVM with degree 2: accuracy 91.35%, training time &#8776 20min, testing_time &#8776 8min
</li>
<li>
Radial basis kernel SVM with default gamma: accuracy 94.46%, training time + testing time &#8776 20min
</li>
<li>
Artificial Neural Network with 1 hidden layer (800-300-10): accuracy 94.46%, training time + testing time &#8776 1hr
</li>
</ul>

<h2>Usage</h2>
