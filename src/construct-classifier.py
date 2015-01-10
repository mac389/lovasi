import random,string
import visualize as artist
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.corpus import stopwords
from scipy.stats import itemfreq

custom_stopwords  = set(stopwords.words('english')  + open('stopwords','rb').read().splitlines())
puncts = set(string.punctuation)

def jaccard(one,two):
	return len(set(one) & set(two))/float(len(set(one) | set(two)))

def similarity(data, flatten = True):
	ans = [[jaccard(data[first_category],data[second_category]) for first_category in categories] 
															 for second_category in categories]
	return ans if not flatten else [item for sublist in ans for item in sublist]

def purify(lst):
	#have a list of strings
	ans = [text.split() for text in lst]
	return [item.lower() for sublist in ans for item in sublist 
		if item not in custom_stopwords and not any([char.isdigit() or char in puncts for char in item])]
data = np.genfromtxt('../data/abstracts-labeled.csv',delimiter = '\t', dtype='str')

categories = np.unique(data[:,1])


compare_groups = True
if compare_groups:
	groups = {category:purify([text[2] for text in data if text[1] == category]) 
			for category in categories}

	fig = plt.figure()
	ax = fig.add_subplot(111)
	sims = similarity(groups, flatten=False)
	cax = ax.imshow(sims, interpolation='nearest',aspect='auto',
		cmap=plt.cm.binary, vmin=0, vmax=1)

	for i,x in enumerate([0.15,0.45,0.8]):
		for j,y in enumerate([0.8,0.45,0.15]):
			if i != j:
				ax.annotate(r'\Large $\mathbf{%.02f}$'%sims[i][j], xy = (x,y), xycoords = 'axes fraction')

	artist.adjust_spines(ax)
	ax.set_yticks(range(3))
	ax.set_yticklabels(map(artist.format,categories))
	
	ax.set_xticks(range(3))
	ax.set_xticklabels(map(artist.format,categories))
	plt.colorbar(cax)
	plt.show()



linear_classification = False
if linear_classification:
	np.random.shuffle(data)

	training,testing = np.array_split(data,2)

	#Extract features
	vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3),strip_accents='unicode')

	X_train = vectorizer.fit_transform(training[:,-1])
	X_test = vectorizer.transform(testing[:,-1])

	print itemfreq(testing[:,1].astype(float))

	nb_classifier = MultinomialNB().fit(X_train,training[:,1].astype(float))

	y_predicted = nb_classifier.predict(X_test)

	print '--------------------------'
	print 'Precision: %.04f'%(metrics.precision_score(testing[:,1].astype(float),y_predicted))
	print 'Recall: %.04f'%(metrics.recall_score(testing[:,1].astype(float),y_predicted))
	print '--------------------------'

	print 'Confusion matrix:'
	print metrics.confusion_matrix(testing[:,1].astype(float),y_predicted)