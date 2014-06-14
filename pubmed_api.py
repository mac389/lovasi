import requests

import matplotlib.pyplot as plt

from Bio.Entrez import efetch, read
from bs4 import BeautifulSoup
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

class PubMedApi(object):
	def __init__(self):
		self.base ='http://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
		self.search = 'esearch.fcgi'
		self.summarize = 'efetch.fcgi'

		self.params = {'db':'pubmed','term':'smoking','retmax':1000}
		self.query = requests.get(self.base+self.search,params=self.params)		

		self.Ids = self.get_ids()
		self.abstracts = {uid[0]:self.fetch_abstract(uid[0]) for uid in self.Ids}

	def get_ids(self):
		text = self.query.text
		soup = BeautifulSoup(text,'xml')
		
		return [id_tag.contents for id_tag in soup.findAll('Id')]

	def __repr__(self):
		return self.query.text

	def get_summaries(self):
		self.summaries = {uid[0]:self.get_abstract(uid[0]) for uid in self.Ids}


	def get_abstract(self,uid):
		self.summaries = requests.get(self.base+self.summarize,params={'id':uid,'db':'pubmed',
												'rettype':'abstract','retmode':'text'})
		text = self.summaries.text
		text = ' '.join(x for x in text.split())
		print text
		return text

	def fetch_abstract(self,pmid):
		handle = efetch(db='pubmed', id=pmid, retmode='xml',email='mac389@gmail.com',retmax=1000)
		xml_data = read(handle)[0]
		try:
		    article = xml_data['MedlineCitation']['Article']	
		    abstract = article['Abstract']['AbstractText'][0]
		    return abstract
		except (IndexError, KeyError):
		    return None


query =PubMedApi()
corpus = filter(None,query.abstracts.values())
print len(corpus)
transformer = TfidfVectorizer(min_df=1)
X = transformer.fit_transform(corpus)

n_components = 5
#Dimensionality reduction with latent semantic analysis
lsa = TruncatedSVD(n_components)
X = lsa.fit_transform(X)
X = Normalizer(copy=False).fit_transform(X)
print X.shape,';;;;;;'

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,0],X[:,1])
ax.set_ylim((-1,1))
ax.set_xlim((-1,1))
plt.tight_layout()
plt.show()
#--Try to cluster PubMed results

'''
for n_cluster in xrange(2,5):
	km = KMeans(n_clusters = n_cluster, init='k-means++',max_iter=100,n_init=1)
	km.fit(X)

	print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, km.labels_,sample_size=1000))

'''