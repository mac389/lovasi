
import sys, gensim, logging, os

import matplotlib.pyplot as plt
import numpy as np
import visualize as artist

from gensim import corpora, models, similarities
from pprint import pprint
from string import punctuation
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, FastICA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from optparse import OptionParser
from matplotlib import rcParams


logging.basicConfig(format='%(message)s', level=logging.INFO, filename='topics')
rcParams['text.usetex'] = True

'''
	Sources:
	    Part of this program is adapted from [http://scikit-learn.org/stable/auto_examples/document_clustering.html#example-document-clustering-py]
'''

#--Command  line parsing
op = OptionParser()
op.add_option('--n', dest='n_components', type='int', help='Preprocess documents with latent semantic analysis')
op.add_option('--analysis_type', dest='analysis_type', type='str', help='Use Gensim for LDA or SKlearn for LSA')
op.print_help()
#

opts,args = op.parse_args()
if len(args) > 0:
	op.error('This script only takes arguments preceded by command line options.')
	sys.exit(1)

filename = 'TF CVD Abstracts.txt'

if opts.analysis_type.lower() == 'lda':
	data = [line.split() for line in open('cleansed_abstracts','rb').read().splitlines()]
	dictionary = corpora.Dictionary(data)
	dictionary.save('./abstracts.dict')

	corpus = [dictionary.doc2bow(text) for text in data]
	corpora.MmCorpus.serialize('./abstracts.mm',corpus)


	lda = models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=100)
	n_topics = 20
	lda.show_topics(topics=20,topn=15)

if opts.analysis_type.lower() == 'lsa':
	data = [line.rstrip() for line in open('cleansed_abstracts','rb').readlines()]
	vectorizer = TfidfVectorizer(ngram_range=(1,3))
	X = vectorizer.fit_transform(data)
	#What about ICA?
	if opts.n_components:
		LSA_DATA = 'abstracts.lsa'
		print 'Reducing dimensionality with LSA'
		lsa = TruncatedSVD(opts.n_components)
		X = lsa.fit_transform(X)
		X = Normalizer(copy=False).fit_transform(X)
		if not os.path.isfile(LSA_DATA):
			np.savetxt(LSA_DATA,X,fmt='%.04f',delimiter ='\t')
	'''
	print 'Performing clustering'
	km = KMeans(n_clusters = 2, init='k-means++',max_iter=100,n_init=1,verbose=True)
	km.fit(X)
	print("Silhouette Coefficient: %0.3f"
	      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
	# N_init refers to the number of replicates, not to the starting cluster number
	# Alternative to KMeans for large samples --> KMeansMiniBatch
	'''

	fig = plt.figure()
	almost_black = '#262626'
	ax = fig.add_subplot(111)
	alphas = (X[:,2]-X[:,2].min())/(X[:,2].max()-X[:,2].min())
	for i,alpha in zip(range(X.shape[0]),alphas):
		ax.scatter(X[i,0],X[i,1],edgecolor=almost_black,alpha=alpha, facecolor='tomato',clip_on=False)
	artist.adjust_spines(ax)
	ax.set_xlabel(artist.format('Principal Component 1'))
	ax.set_ylabel(artist.format('Principal Component 2'))
	ax.set_ylim((-1,1))
	ax.set_xlim((-1,1))
	plt.show()