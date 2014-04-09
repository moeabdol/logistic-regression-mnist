import theano
import numpy
import theano.tensor as T
import matplotlib.pyplot as plt
from matplotlib import *
from Utils import *
from Logistic_Regression import *

learning_rate = 0.13
n_epochs = 1000
dataset = 'mnist.pkl.gz'
batch_size = 600

# load datasets
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for all sets
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

###################
# build the model #
###################
print '... building the model'
index = T.lscalar()	# index to minibatch
x = T.matrix('x')	# data of rasterized images
y = T.ivector('y')	# labels are 1D vector of int
classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)	# logistic regression classifier
cost = classifier.negative_log_likelihood(y)	# cost to minimize during training

test_model = theano.function(inputs=[index],
		outputs=classifier.errors(y),
		givens={x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]})

validate_model = theano.function(inputs=[index],
		outputs=classifier.errors(y),
		givens={x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

# model to show misclassified examples
misclassified_model = theano.function(inputs=[index],
		outputs=classifier.errors(y),
		givens={x: test_set_x[index: (index + 1)],
			y: test_set_y[index: (index + 1)]})

# compute the gradient of cost with respect to theta = (W, b)
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

# specify how to update the parameters of the model
updates = [(classifier.W, classifier.W - learning_rate * g_W),
	(classifier.b, classifier.b - learning_rate * g_b)]

train_model = theano.function(inputs=[index],
		outputs=cost,
		updates=updates,
		givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]})

###############
# train model #
###############
def train():
	print '... training the model'
	epoch = 0
	while(epoch < n_epochs):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
		print 'training epoch %i' %epoch

def test():
	for minibatch_index in xrange(n_test_batches):
		minibatch_mean_error = test_model(minibatch_index)
		print 'testing minibatch %i and mean error = %f' %(minibatch_index, minibatch_mean_error)

def show_misclassified():
	for i in xrange(n_test_batches * batch_size):
		error = misclassified_model(i)
		if error == 1:
			print 'misclassified example found in test set at index %i' %i
			image = test_set_x[i].eval()
			label = test_set_y[i].eval()

			z = T.vector('z')
			classifier.p_y_given_x = T.nnet.softmax(T.dot(z, classifier.W) + classifier.b)
			classifier.y_pred = T.argmax(classifier.p_y_given_x, axis=1)
			misclassified_label = theano.function(inputs=[z], outputs=classifier.y_pred)
			mis_label = misclassified_label(image)

			# show actual image
			image = image.reshape(28, 28)
			plt.imshow(image, cmap=cm.gray)
			plt.xlabel('Actual ' + str(label))
			plt.ylabel('Prediction ' + str(mis_label[0]))
			plt.show()






















