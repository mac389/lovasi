import numpy as np 
import matplotlib.pyplot as plt 
import Graphics as artist
import matplotlib.gridspec as gridspec

from matplotlib import rcParams
from awesome_print import ap 

rcParams['text.usetex'] = True
def unique_words(aStr):
	return ' '.join([word for word in set(aStr.split())])

def princomp(A,numpc=3):
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
	[latent,coeff] = np.linalg.eig(np.cov(M))
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

TEXT = 1
basis_vectors = [unique_words(line.split(':')[TEXT]) for line in open('../data/lda-topics.txt','rb').read().splitlines()]

def jaccard_similarity(a,b):
	a = set(a)
	b = set(b)

	try:
		return len(a & b)/float(len(a | b))
	except: 
		return 0 

#ap(basis_vectors)

basis_vectors_similarity = np.array([[jaccard_similarity(one,two) for one in basis_vectors] for two in basis_vectors])

eigvecs,proj,eigvals = princomp(basis_vectors_similarity,numpc=basis_vectors_similarity.shape[1])

for name,matrix in [('eigvecs',eigvecs),('projections',proj),('eigvals',eigvals)]:
	np.savetxt('../data/%s'%name,matrix,fmt='%.02f')

max_color = max(basis_vectors_similarity.max(),eigvecs.max(),np.corrcoef(eigvecs.T).max())
min_color = min(basis_vectors_similarity.min(),eigvecs.min(),np.corrcoef(eigvecs.T).max())

fig = plt.figure(figsize=(12,5))
gspec = gridspec.GridSpec(1, 3,width_ratios=[1,1,1])
non_orthogonal = plt.subplot(gspec[0])
loading_matrix = plt.subplot(gspec[2])
orthogonal = plt.subplot(gspec[1])

cno = non_orthogonal.imshow(basis_vectors_similarity,interpolation='nearest',aspect='auto',vmax=max_color,vmin=min_color)
artist.adjust_spines(non_orthogonal)
non_orthogonal.set_xlabel(artist.format('Topic'))
non_orthogonal.set_ylabel(artist.format('Topic'))
cbar_no = fig.colorbar(cno,ax=non_orthogonal)
cbar_no.set_label(artist.format('Jaccard Similarity '))
cax_load = loading_matrix.imshow(eigvecs,interpolation='nearest',aspect='auto',vmax=max_color,vmin=min_color)
artist.adjust_spines(loading_matrix)

loading_matrix.set_xlabel(artist.format('Eigenvector'))
loading_matrix.set_ylabel(artist.format('Topic'))
cbar_load = fig.colorbar(cax_load,ax=loading_matrix,use_gridspec=True)
cbar_load.set_label(artist.format('Loading Weight'))

cax = orthogonal.imshow(np.corrcoef(eigvecs.T),interpolation='nearest',aspect='auto',vmax=max_color,vmin=min_color)
artist.adjust_spines(orthogonal)
orthogonal.set_xlabel(artist.format('Eigenvector'))
orthogonal.set_ylabel(artist.format('Topic'))

cbar = fig.colorbar(cax,ax=orthogonal,use_gridspec=True)
cbar.set_label(artist.format('Correlation Coefficient'))
gspec.tight_layout(fig)
plt.show()