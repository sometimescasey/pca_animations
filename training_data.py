import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
import matplotlib.gridspec as gridspec

# force all grayscale images to use consistent 0 -> 1 as min/max
VMIN = -1.0
VMAX = 1.0

s = 2
digits = ["check", "bar"]
class_map = {0:"check", 1:"bar"}
IMG_PATH = "2by2/"
BASES = [1, 2, 3, 4]
PROJ_IMG_PATHS = [ IMG_PATH + "proj_test/p" + str(i) + ".png" for i in BASES ]

# s = 3
# digits = [0, 1, 4, 7]
# class_map = {0:0, 1:1, 2:4, 3:7}
# IMG_PATH = "nums/"
# BASES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# PROJ_IMG_PATHS = [ IMG_PATH + "proj_test/p" + str(i) + ".png" for i in BASES ]

k = len(digits) # number of classes, 4
COUNT = 100
NOISE_SCALE = 255
DENOM = 255

EXT = ".png"

def generate_data(img_path):
	data = np.empty((COUNT, k, s*s))

	j = 0
	for digit in digits:
		print("j is {}, digit is {}".format(j,digit))
		img = np.array(Image.open(img_path + str(digit) + EXT).convert('L'))
		if not os.path.exists(img_path + str(digit)):
			os.makedirs(img_path + str(digit))

		w = img.shape[0]
		h = img.shape[1]

		for i in range(COUNT):
			intensity_noise = (np.random.rand(1) * 1.5) + 0.5 # scale from 0.5 to 2x what it was before
			background_noise = (np.random.rand(w,h) - 0.5) * NOISE_SCALE 
			# noise = 0
			noisy_img = (img * intensity_noise) + background_noise
			rescaled = noisy_img / DENOM # force range 0 to 1
			data[i,j] = rescaled.reshape((s*s))
			save_noisy = Image.fromarray(noisy_img).convert('L')
			save_noisy.save(img_path + str(digit) + "/" + str(digit) + "_" + str(i) + EXT)

		j += 1

	return data

def train_test_split(train_ratio=0.7):
	data = generate_data(IMG_PATH)

	# train test split
	r_ratio = train_ratio
	t_ratio = 1 - r_ratio

	r_index = int(r_ratio*COUNT)

	train_data = data[0:r_index,:,:]
	test_data = data[r_index:COUNT,:,:]

	return train_data, test_data

def open_img_flat(img_path):
	return np.array(Image.open(img_path).convert('L')).reshape((s*s)) / DENOM

def open_img(img_path):
	return np.array(Image.open(img_path).convert('L')) / DENOM

def remove_ticks(s_ax):
	# convenience fxn: remove tickmarks from subplot axes
	s_ax.tick_params(       
		which='both',      # remov ticks and labels
		labelbottom=False,
		labelleft=False,
		length=0.0)

def get_means(data):
	means = np.zeros((k, s*s))

	for c in range(k):
		class_subarray = data[:,c,:]
		means[c] = np.average(class_subarray, axis=0)

	return means

def display_means(means):
	# Neat visual! 	
	fig, ax = plt.subplots(nrows=1, ncols=k)
	fig.suptitle("Means for each class")
	for c in range(k):
		reshaped = np.asarray(np.reshape(means[c], (s, s)))
		s_ax = plt.subplot(1, k, c+1)
		remove_ticks(s_ax)
		plt.title("class " + str(class_map[c]))
		plt.imshow(reshaped, cmap='gray_r')
	plt.show()

def compute_sigmas(data, means):

	covariances = np.zeros((k, s*s, s*s))

	# for stability
	diag = np.identity(s*s) * 0.01
	
	# # compute average of (point - mean for class)^2 for each class
	for c in range(k):
		class_subarray = data[:,c,:]
		residuals = (class_subarray - means[c])
		sum_sq_residuals = np.dot(residuals.T, residuals)
		covariances[c] = sum_sq_residuals / class_subarray.shape[0] + diag
		# covariances[c] = np.cov(class_subarray, rowvar=False) + diag

	return covariances

def plot_eigenvectors(covariances):
	fig, ax = plt.subplots(nrows=1, ncols=k)
	fig.suptitle("Top Eigenvector for each class")
	for c in range(k):

		eigenvalues, eigenvectors = np.linalg.eig(covariances[c])
		print(eigenvalues)
		print("covariances[c] shape: {}".format(covariances[c].shape))

		biggest_index = eigenvalues.argmax() # should be the first one, but just in case
		print("biggest_index: {}".format(biggest_index))

		reshaped = np.asarray(np.reshape(eigenvectors[:,biggest_index], (s, s)))
		s_ax = plt.subplot(1, k, c+1)
		remove_ticks(s_ax)

		plt.title("class " + str(class_map[c]) + "\n" + r"$\lambda$" + " = " + str(round(eigenvalues[c], 3)))
		plt.imshow(reshaped, cmap='gray_r')
	plt.show()

def generative_likelihood(digits, means, covariances):

	D = digits.shape[1]
	N = digits.shape[0]

	gen_loglik = np.zeros((k, N))

	for c in range(k):
		residuals = np.subtract(digits, means[c])

		sigma_i = np.linalg.inv(covariances[c])

		sq_mahalanobis = np.einsum('ij,ij->i', np.matmul(residuals, sigma_i), residuals)
		# Note that this einsum is element-wise dot product: more efficient way of doing matmul and taking the diagonal

		pi_bit = -0.5 * D *np.log(2 * np.pi)

		det_bit = -0.5 * np.log(np.linalg.det(covariances[c]))

		maha_bit = -0.5 * sq_mahalanobis

		gen_loglik[c] = pi_bit + det_bit + maha_bit

	return gen_loglik

def conditional_likelihood(digits, means, covariances):

	gen_loglik = generative_likelihood(digits, means, covariances)

	p_y = 1.0/k # uniform prior
	p_xs = logsumexp(gen_loglik)

	cond_loglik = gen_loglik.T + np.log(p_y) - p_xs # log already applied to p_xs
	
	return cond_loglik

def classify_data(digits, means, covariances):
	'''
	Classify new points by taking the most likely posterior class
	'''
	cond_likelihood = conditional_likelihood(digits, means, covariances)
	
	N = digits.shape[0]

	pred = cond_likelihood.argmax(axis=1)
	float_preds = np.array(pred).astype(float)

	return float_preds

def do_basis_projections(input_img, covariances, mtx):

	cmap_setting = 'gray_r'

	n = k # number of classes

	fig, ax = plt.subplots(nrows=1, ncols=n+1) # +1 to display the original image as well

	fig.suptitle("Unit bases projected against covariance matrices")
	fig.set_figheight(10)
	fig.set_figwidth(10)

	s_ax = plt.subplot(1, n+1, 1)
	remove_ticks(s_ax)

	plt.title("Original Unit Basis")
	# plt.imshow(input_img.reshape((s,s)), cmap=cmap_setting)
	sq_proj = input_img.reshape((s,s))
	s_ax.matshow(sq_proj, cmap=cmap_setting)
	for (i,j),z in np.ndenumerate(sq_proj):
		s_ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=8, color='red')

	for c in range(k):
	# Apply cov matrix for each class, show result
		projection = np.dot(covariances[c], input_img.reshape(s*s))
		s_ax = plt.subplot(1, n+1, c+2)
		remove_ticks(s_ax)

		plt.title("class " + str(class_map[c]))
		# plt.imshow(projection.reshape((s,s)), cmap=cmap_setting)
		sq_proj = projection.reshape((s,s))
		s_ax.matshow(sq_proj, cmap=cmap_setting)
		for (i,j),z in np.ndenumerate(sq_proj):
			s_ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=8, color='red')

	plt.show()

def plot_pca_grid(train_data, covariances):
	classes = range(k)
	k_max = len(classes)
	dims = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	d_max = len(dims)

	fig, ax = plt.subplots(nrows=k_max, ncols=d_max)
	fig.suptitle("Projection and reconstruction onto N basis vectors")
	fig.set_figheight(10)
	fig.set_figwidth(10)

	count = 1
	for c in classes:
		for d in dims:
			mean_proj = project_onto_basis(count, train_data, 
				covariances, c, k_max, d, d_max)
			reshaped = np.asarray(np.reshape(mean_proj, (s, s)))
			s_ax = plt.subplot(k_max, d_max, count)
			remove_ticks(s_ax)
			if c == 0:
				# put labels on first row
				plt.title("{}".format(d))
			plt.imshow(reshaped, cmap='gray_r')
			count += 1

	fig.tight_layout()
	plt.show()

def project_onto_basis(grid_count, train_data, covariances, c, k_max, d, d_max):
	mean_proj = np.zeros((1, s*s))

	eigenvalues, _eigenvectors = np.linalg.eig(covariances[c])
	eigenvectors = _eigenvectors.T

	in_order = eigenvalues.argsort()
	sorted_eigenvalues = np.flip(eigenvalues[in_order], axis=0)
	sorted_eigenvectors = np.flip(eigenvectors[in_order], axis=0)

	# get top d eigenvectors
	new_basis = sorted_eigenvectors[:d] # each row is an eigenvector
	
	# note that this new_basis is orthonormal (orthogonal, all vectors normalized to unit length)
	# so new_basis inverse = new_basis transpose

	class_subarray = train_data[:,c,:]
	projection = np.dot(new_basis, class_subarray.T)

	# project back
	reverse = np.dot(new_basis.T, projection)

	# average all projections
	mean_proj = np.average(reverse.T, axis=0)

	return mean_proj

def evaluate_GDA(train_data, test_data, means, covariances):
	# Evaluate our Gaussian Discriminant Analysis classifier

	train_accuracy = 0.0
	for i in range(k):
		train_preds = classify_data(train_data[:,i,:], means, covariances)
		train_accuracy += np.sum(np.equal(train_preds, i)) / train_data[:,i,:].shape[0]
	print("train accuracy: {}".format(train_accuracy / k))

	test_accuracy = 0.0
	for i in range(k):
		test_preds = classify_data(test_data[:,i,:], means, covariances)
		test_accuracy += np.sum(np.equal(test_preds, i)) / test_data[:,i,:].shape[0]
	print("test accuracy: {}".format(test_accuracy / k))

def main():

	train_data, test_data = train_test_split(train_ratio=0.7)

	means = get_means(train_data)
	display_means(means)

	covariances = compute_sigmas(train_data, means)
	plot_eigenvectors(covariances)

	for path in PROJ_IMG_PATHS:
	 	input_img = open_img_flat(path)
	 	mtx = open_img(path)
	 	print(mtx)
	 	do_basis_projections(input_img, covariances, mtx)

	# plot_pca_grid(train_data, covariances)

	# evaluate_GDA(train_data, test_data, means, covariances)

if __name__ == '__main__':
	main()




