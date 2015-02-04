import string, os, itertools 

import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist
import matplotlib.gridspec as gridspec

from awesome_print import ap
from matplotlib import rcParams
from sklearn.decomposition import PCA,RandomizedPCA 
rcParams['text.usetex'] = True

def unique_words(aStr):
	return ' '.join([word for word in set(aStr.split())])

TEXT = 1
punkt = set(string.punctuation)
basis_vectors = [unique_words(line.split(':')[TEXT]) for line in open('../data/lda-topics.txt','rb').read().splitlines()]
stopwords = set(open('stopwords').read().splitlines())
data = [line.lower() for line in open('../data/cleansed_abstracts','rb').read().splitlines()]
data = [' '.join([''.join(ch for ch in word if ord(ch)<128) for word in line.split() 
		if word not in stopwords and 'file://' not in word and not any([ch in punkt for ch in word])]).split() for line in data]
#data = list(itertools.chain.from_iterable(data))

def jaccard_similarity(a,b):
	a = set(a)
	b = set(b)

	try:
		return len(a & b)/float(len(a | b))
	except: 
		return 0 

def princomp(A,numpc=3):
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
	[latent,coeff] = np.linalg.eig(cov(M))
	p = np.size(coeff,axis=1)
	idx = np.argsort(latent) # sorting the eigenvalues
	idx = idx[::-1]       # in ascending order
	# sorting eigenvectors according to the sorted eigenvalues
	coeff = coeff[:,idx]
	latent = latent[idx] # sorting eigenvalues
	if numpc < p and numpc >= 0:
		coeff = coeff[:,range(numpc)] # cutting some PCs if needed
	score = np.dot(coeff.T,M) # projection of the data in the new space
	return coeff,score,latent

def gs(X, row_vecs=True, norm = True):
	if not row_vecs:
		X = X.T
	Y = X[0:1,:].copy()
	for i in range(1, X.shape[0]):
		proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
		Y = np.vstack((Y, X[i,:] - proj.sum(0)))
	if norm:
		Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
	if row_vecs:
		return Y
	else:
		return Y.T

#Need to use compressed formats
if not os.path.isfile('projection_data_nonorthogonal_vectors'):
	similarity_matrix = np.array([[jaccard_similarity(vector,entry) for vector in basis_vectors] for entry in data])
	np.savetxt('projection_data_nonorthogonal_vectors',similarity_matrix,fmt='%.04f',delimiter='\t')
else:
	similarity_matrix = np.loadtxt('projection_data_nonorthogonal_vectors',delimiter='\t')

#Orthogonalize basis_vectors, easier to use PCA of JS of basis_vectors
if not os.path.isfile('basis_vector_correlation_matrix'):
	basis_vector_correlation_matrix = np.array([[jaccard_similarity(one,two) for one in basis_vectors] for two in basis_vectors])
	np.savetxt('basis_vector_correlation_matrix',basis_vector_correlation_matrix,fmt='%.04f',delimiter='\t')
else:
	basis_vector_correlation_matrix = np.loadtxt('basis_vector_correlation_matrix',delimiter='\t')

if not os.path.isfile('data_correlation_matrix.npz'):
	data_correlation_matrix = np.zeros((len(data),(len(data))))
	for i in xrange(len(data)):
		for j in xrange(i-1):
			print i,j,len(data)
			data_correlation_matrix[i,j] = jaccard_similarity(data[i],data[j])

	data_correlation_matrix += data_correlation_matrix.T
	data_correlation_matrix[np.diag_indices_from(data_correlation_matrix)] = 1
	np.savez_compressed('data_correlation_matrix',data_correlation_matrix=data_correlation_matrix)
else:
	data_correlation_matrix = np.load('data_correlation_matrix.npz')['data_correlation_matrix']

pca = RandomizedPCA(n_components=3)
for name,dataset in [('data-correlation-matrix',data_correlation_matrix),('basis-vector-correlation-matrix',basis_vector_correlation_matrix)]:
	print name
	dataset_r = pca.fit(dataset).transform(dataset)
	print pca.explained_variance_ratio_
	np.savetxt('%s-reduced'%(name),dataset_r,fmt='%.04f')



#ap(gs(basis_vector_correlation_matrix))
gspec = gridspec.GridSpec(1, 3,width_ratios=[5,5,1],height_ratios=[1])
left = plt.subplot(gspec[0])
right = plt.subplot(gspec[1])
scale = plt.subplot(gspec[2])


orthogonalized_basis_vector_correlation_matrix = gs(basis_vector_correlation_matrix)*gs(basis_vector_correlation_matrix)
vmin = min(basis_vector_correlation_matrix.max(),orthogonalized_basis_vector_correlation_matrix.min())
vmax = max(basis_vector_correlation_matrix.max(),orthogonalized_basis_vector_correlation_matrix.max())
lax = left.imshow(basis_vector_correlation_matrix,interpolation='nearest',aspect='auto',vmin=vmin,vmax=vmax)
rax = right.imshow(orthogonalized_basis_vector_correlation_matrix,interpolation='nearest',aspect='auto',vmin=vmin,vmax = vmax)
cbar = plt.colorbar(rax,cax=scale)
map(artist.adjust_spines,[left,right])
left.set_ylabel(artist.format('Topic'))
left.set_xlabel(artist.format('Topic'))
left.set_title(artist.format('LDA Topics'))
right.set_title(artist.format('Eigenvectors of LDA Topics'))
right.set_ylabel(artist.format('Topic'))
right.set_xlabel(artist.format('Topic'))
cbar.set_label(artist.format('Jaccard Similarity'))
cbar.set_clim((0,1))
plt.tight_layout()
plt.show()

'''
fig = plt.figure()
ax = fig.add_subplot(111)
#cax = ax.imshow(basis_vector_correlation_matrix,interpolation='nearest',aspect='auto')
cax = ax.imshow(similarity_matrix,interpolation='nearest',aspect='auto')
artist.adjust_spines(ax)
cbar = plt.colorbar(cax)
cbar.set_label(artist.format('Jaccard similarity'))
ax.set_ylabel(artist.format('PIs'))
ax.set_xlabel(artist.format('LDA topics'))
plt.tight_layout()
plt.show()
'''
