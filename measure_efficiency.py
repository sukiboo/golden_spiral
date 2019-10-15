
from scipy.stats.stats import pearsonr
from scipy.stats.mstats import gmean
import numpy as np


def sequence_correlation(seq):

	# compute correlation coefficient between every pair of sequences
	cor = 0
	for n in range(seq.shape[0]):
		for k in range(n+1, seq.shape[0]):
			cor = max(cor, np.abs(pearsonr(seq[n],seq[k])[0]))

	print('maximum correlation: {:.6f}'.format(cor))

	return


def sequence_discrepancy(seq, num_tests=10000):

	d, num_points = seq.shape
	a = np.linspace(0, 1, num_tests+1)[1:]
	D = np.zeros(num_tests)

	for n in range(num_tests):
		D[n] = np.abs(np.sum(np.prod((seq < a[n]), axis=0)) / num_points - a[n]**d)

	print('D*-discrepancy: {:.6f}'.format(np.max(D)))

	return


# compute the matrix of distances between the points
def dist_matrix(s):
	n = s.shape[0]
	dst = np.zeros((n,n)) - 1
	for n in range(n):
		dst[n] = np.linalg.norm(s - s[n], axis=1)
	dst.sort(axis=1)
	return dst


# measure points density
def density(dim, obj, s, dst, num_measures=10):

	# extract the number of points
	num_points = s.shape[0]
	# take num_measures radius values
	r = np.linspace(0, 1, num_measures+1)[1:] * np.pi/2

	# estimate density
	dns = np.zeros((num_points,num_measures), dtype=np.int32)
	for k in range(num_measures):
		# find the number of points in a ball of radius r[k]
		dns[:,k] = np.count_nonzero(np.where(dst < r[k], dst, 0), axis=1)
		# adjust for balls
		if obj=='ball':
			scale = 1 - np.clip((1 - np.linalg.norm(s, axis=1))/r[k], 0, 1)
			scale = 1 - scale**dim / 2
			dns[:,k] = dns[:,k] / scale

	return dns


def measure_coverage(dim, obj, s, dst, num_rnd=1000):

	# estimate average distance to the closest neighbor
	dst_cl_avg = np.mean(dst[:,1])

	# generate num_rnd random points
	d = (dim if obj=='ball' else dim+1)
	s_rnd = np.random.randn(num_rnd, d)
	s_rnd /= np.linalg.norm(s_rnd, axis=1).reshape((-1,1))
	if obj in ['ball','b']:
		s_rnd *= (np.random.uniform(0,1,(num_rnd,1)))**(1/d)

	# find distances to the dim-th closest point
	dst_rnd = np.zeros(num_rnd)
	for k in range(num_rnd):
		dst_rnd[k] = np.partition(np.linalg.norm(s-s_rnd[k], axis=1), dim)[dim]

	# find maximal distance to the dim-th closest point
	dst_rnd_max = np.max(dst_rnd)
	# compute ratio
	cvr_ratio = dst_cl_avg / dst_rnd_max

	return cvr_ratio


# compute and display various stats
def display_stats(dim, obj, s):

	num_points = s.shape[0]

	# compute the distance matrix
	dst = dist_matrix(s)
	# average distances (this only makes sense for a sphere)
	dst_avg = dst.mean(axis=0, keepdims=True)
	dst_std = np.std(np.linalg.norm(dst - dst_avg, axis=1))
	# average distance to the closest neighbor ratio
	dst_cl = np.sort(dst[:,1])
	num_avg = np.floor(dst_cl.size / 10).astype(int)
	dst_cl_ratio = np.mean(dst_cl[:num_avg]) / np.mean(dst_cl[-num_avg:])

	# measure and normalize density
	dns = density(dim, obj, s, dst)
	# #dns = dns / (np.max(dns, axis=0) + 1e-16)
	# compute standard deviation of dns
	dns_std = np.std(dns, axis=0)
	# adjust dns_std to account for the number of points, radius, and dimensionality
	dns_std = dns_std / num_points# / (np.linspace(0, 1, 11)[1:])**dim
	# arithmetic and geometric means of dns_std
	dns_std_am = np.mean(dns_std, axis=0)
	# #dns_std_gm = gmean(dns_std, axis=0)

	# measure coverage
	cvr_ratio = measure_coverage(dim, obj, s, dst, num_rnd=10000)


	print()
	print('efficiency stats, smaller is better:')
	# #print('  distances deviation: {:.6f}'.format(dst_std))
	print('  density deviation  : {0}'.format(dns_std))
	# #print('  density gmean      : {:.4f}'.format(dns_std_gm))
	print('  density mean       : {:.4f}'.format(dns_std_am))
	print('  1 - min_dist_ratio : {:.4f}'.format(1 - dst_cl_ratio))
	print('  1 - coverage_ratio : {:.4f}'.format(1 - cvr_ratio))
	print()

	stats = [dns_std_am, 1 - dst_cl_ratio, 1 - cvr_ratio]

	return stats
