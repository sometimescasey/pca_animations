'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.

Casey Juanxi Li
998816973
'''

import data
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
from PIL import Image, ImageFilter

def compute_mean_mles(train_data, train_labels):
	# Compute the mean estimate for each digit class

	# Should return a numpy array of size (10,64)
	# The ith row will correspond to the mean estimate for digit class i

	means = np.zeros((10, 64))

	for k in range(10):
		class_subarray = train_data[np.nonzero(train_labels == k)]
		means[k] = np.average(class_subarray, axis=0)

	return means

def display_mean_mles(means):
	# Neat visual! 	
	fig, ax = plt.subplots(nrows=2, ncols=5)

	for q in range(10):
		reshaped = np.asarray(np.reshape(means[q], (8, 8)))
		plt.subplot(2, 5, q+1)
		plt.title(q)
		plt.imshow(reshaped, cmap='gray_r')
	plt.show()

def remove_ticks(s_ax):
	# convenience fxn: remove tickmarks from subplot axes
	s_ax.tick_params(       
		which='both',      # remov ticks and labels
		labelbottom=False,
		labelleft=False,
		length=0.0)

def one_per_class(train_data, train_labels):
	fig, ax = plt.subplots(nrows=2, ncols=5)
	for k in range(10):
		class_subarray = train_data[np.nonzero(train_labels == k)]
		reshaped = class_subarray[0].reshape((8,8))
		s_ax = plt.subplot(2, 5, k+1)
		remove_ticks(s_ax)
		plt.imshow(reshaped, cmap='gray_r')
	plt.show()

def one_class(train_data, train_labels, klass):
	fig, ax = plt.subplots(nrows=2, ncols=5)
	c = float(klass)
	for i in range(10):
		class_subarray = train_data[np.nonzero(train_labels == c)]
		reshaped = class_subarray[i].reshape((8,8))
		s_ax = plt.subplot(2, 5, i+1)
		remove_ticks(s_ax)
		plt.imshow(reshaped, cmap='gray_r')
	plt.show()

def one_class_save(train_data, train_labels, klass):
	c = float(klass)
	for i in range(10):
		class_subarray = train_data[np.nonzero(train_labels == c)]
		reshaped = 255 - (class_subarray[i].reshape((8,8)) * 255)
		save_img = Image.fromarray(reshaped).convert('L')
		save_img.save(str(c) + "_" + str(i) + ".png")

def display_img_w_label(digits, labels):
	# display a few images with their labels
	N = digits.shape[0]
	fig, ax = plt.subplots(nrows=1, ncols=N)
	for q in range(N):
		reshaped = np.asarray(np.reshape(digits[q], (8, 8)))
		plt.subplot(1, N, q+1)
		plt.title(labels[q])
		plt.imshow(reshaped, cmap='gray_r')
	plt.show()

def compute_sigma_mles(train_data, train_labels):
	'''
	Compute the covariance estimate for each digit class

	Should return a three dimensional numpy array of shape (10, 64, 64)
	consisting of a covariance matrix for each digit class

	'''
	means = compute_mean_mles(train_data, train_labels)

	covariances = np.zeros((10, 64, 64))

	# create a 64x64 and put 0.01I along the diagonal
	diag = np.identity(64) * 0.01
	
	# # compute average of (point - mean for class)^2 for each class
	for k in range(10):
		class_subarray = train_data[np.nonzero(train_labels == k)]
		residuals = (class_subarray - means[k])
		sum_sq_residuals = np.dot(residuals.T, residuals)
		covariances[k] = sum_sq_residuals / class_subarray.shape[0] + diag
		# covariances[k] = np.cov(class_subarray, rowvar=False) + diag

	return covariances

def plot_eigenvectors(covariances):
	fig, ax = plt.subplots(nrows=2, ncols=5)
	for k in range(10):

		eigenvalues, _eigenvectors = np.linalg.eig(covariances[k])
		eigenvectors = _eigenvectors.T

		biggest_index = eigenvalues.argmax() # should be the first one, but just in case

		reshaped = np.asarray(np.reshape(eigenvectors[biggest_index], (8, 8)))
		plt.subplot(2, 5, k+1)
		plt.imshow(reshaped, cmap='gray_r')
	plt.show()

def plot_pca_grid(train_data, train_labels, covariances, means):
	classes = range(10)
	k_max = len(classes)
	dims = [1, 2, 4, 8, 16, 32, 64]
	d_max = len(dims) + 1 # +1 for original

	fig, ax = plt.subplots(nrows=k_max, ncols=d_max)
	fig.set_figheight(15)
	fig.set_figwidth(15)

	count = 1
	for k in classes:
		for d in dims:
			mean_proj = project_onto_basis(count, train_data, train_labels, 
				covariances, k, k_max, d, d_max)
			reshaped = np.asarray(np.reshape(mean_proj, (8, 8)))
			plt.subplot(k_max, d_max, count)
			if k == 0:
				# put labels on first row
				plt.title("Basis eigenvectors = {}".format(d))
			plt.imshow(reshaped, cmap='gray_r')
			count += 1
		
		plt.subplot(k_max, d_max, count)
		if k == 0:
			plt.title("Original".format(d))
		plt.imshow(np.reshape(means[k], (8,8)), cmap='gray_r')
		count += 1
			

	fig.tight_layout()
	plt.show()

def project_onto_basis(grid_count, train_data, train_labels, covariances, k, k_max, d, d_max):
	mean_proj = np.zeros((1, 64))

	eigenvalues, _eigenvectors = np.linalg.eig(covariances[k])
	eigenvectors = _eigenvectors.T

	in_order = eigenvalues.argsort()
	sorted_eigenvalues = np.flip(eigenvalues[in_order], axis=0)
	sorted_eigenvectors = np.flip(eigenvectors[in_order], axis=0)

	# get top d eigenvectors
	new_basis = sorted_eigenvectors[:d] # each row is an eigenvector
	
	# note that this new_basis is orthonormal (orthogonal, all vectors normalized to unit length)
	# so new_basis inverse = new_basis transpose

	class_subarray = train_data[np.nonzero(train_labels == k)]
	projection = np.dot(new_basis, class_subarray.T)

	# project back
	reverse = np.dot(new_basis.T, projection)

	# average all projections
	mean_proj = np.average(reverse.T, axis=0)

	return mean_proj

def generative_likelihood(digits, means, covariances):

	D = digits.shape[1]
	N = digits.shape[0]

	'''
	Compute the generative log-likelihood:
	    log p(x|y,mu,Sigma)

	Should return an n x 10 numpy array 

	This is the log-lik of the data given y (class). This is the class-dependent Gaussian PDF.

	-d/2 log(2 * pi) - 1/2 log (det(sigma)) - 1/2((x-u_k).T sigma_inverse (x-u_k)) 

	Note that this is a 64D Gaussian: one image has 64 pixels, so a 64 x 1 mean is defined for each of 10 classes

	'''

	gen_loglik = np.zeros((10, N))
	gen_loglik2 = np.zeros((10, N, N))

	for k in range(10):
		residuals = np.subtract(digits, means[k])

		sigma_i = np.linalg.inv(covariances[k])

		sq_mahalanobis = np.einsum('ij,ij->i', np.matmul(residuals, sigma_i), residuals)
		# Note that this einsum is element-wise dot product: more efficient way of doing matmul and taking the diagonal

		pi_bit = -0.5 * D *np.log(2 * np.pi)

		det_bit = -0.5 * np.log(np.linalg.det(covariances[k]))

		maha_bit = -0.5 * sq_mahalanobis

		gen_loglik[k] = pi_bit + det_bit + maha_bit

	return gen_loglik

def conditional_likelihood(digits, means, covariances):
	'''
	Compute the conditional likelihood:

	    log p(y|x, mu, Sigma)

	This should be a numpy array of shape (n, 10)
	Where n is the number of datapoints and 10 corresponds to each digit class

	This is log-lik of label given the data, or the expression from lec 14 slide 30 (minus the typo)

	'''

	gen_loglik = generative_likelihood(digits, means, covariances)

	p_y = 1.0/10 # prior given in assignment
	p_xs = logsumexp(gen_loglik)

	cond_loglik = gen_loglik.T + np.log(p_y) - p_xs # log already applied to p_xs
	
	return cond_loglik

def avg_conditional_likelihood(digits, labels, means, covariances):
	'''
	Compute the average conditional likelihood over the true class labels

	    AVG( log p(y_i|x_i, mu, Sigma) )

	i.e. the average log likelihood that the model assigns to the correct class label
	'''

	# Assuming this means that given N digits to classify,
	# grab the log-lik of the correct label given the data from each
	# sum them all together, and divide by N. 
	# Return just one number.

	cond_likelihood = conditional_likelihood(digits, means, covariances)

	N = digits.shape[0]

	correct_label_logliks = np.zeros((N,1))

	# for each N, grab the loglik of the correct class from each row
	# how to vectorize this??
	for i in range(N):
		correct_label_logliks[i] = cond_likelihood[i][int(labels[i])]

	avg = np.average(correct_label_logliks)

	return avg

def classify_data(digits, means, covariances):
	'''
	Classify new points by taking the most likely posterior class
	'''
	cond_likelihood = conditional_likelihood(digits, means, covariances)
	
	N = digits.shape[0]

	pred = cond_likelihood.argmax(axis=1)
	float_preds = np.array(pred).astype(float)

	return float_preds

def main():

	### For first run only
	# train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('hw5digits.zip', './data')

	train_data, train_labels, test_data, test_labels = data.load_all_data('data', shuffle=True)

	# one_per_class(train_data, train_labels)
	for j in range(10):
		one_class_save(train_data, train_labels, j)
	### [CHECK: DATA] Display some images and labels to ensure those match up
	n = 20
	display_img_w_label(train_data[0:n], train_labels[0:n])
	
	# Fit the model
	means = compute_mean_mles(train_data, train_labels)
	
	### [CHECK: MEANS] Display means to make sure they are sensible
	display_mean_mles(means)
	
	covariances = compute_sigma_mles(train_data, train_labels)
	
	### [ CHECK: COVARIANCE, EIGENVECTORS ] Plot some PCA projections to verify
	plot_pca_grid(train_data, train_labels, covariances, means)

	# Q 1a)
	train_avg = avg_conditional_likelihood(train_data, train_labels, means, covariances)
	print("avg conditional loglik, train: {}".format(train_avg))

	test_avg = avg_conditional_likelihood(test_data, test_labels, means, covariances)
	print("avg conditional loglik, test: {}".format(test_avg))

	# Evaluation, Q 1b)
	train_preds = classify_data(train_data, means, covariances)
	train_accuracy = np.sum(np.equal(train_preds, train_labels))/train_data.shape[0]
	print("train accuracy: {}".format(train_accuracy))

	test_preds = classify_data(test_data, means, covariances)
	test_accuracy = np.sum(np.equal(test_preds, test_labels))/test_data.shape[0]
	print("test accuracy: {}".format(test_accuracy))

	# Q 1c)
	plot_eigenvectors(covariances)

if __name__ == '__main__':
	main()
