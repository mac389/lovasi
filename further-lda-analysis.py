import os

import numpy as np
import visualize as artist
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['text.usetex'] = True

def jaccard_index(abstract,topic):
	abstract = set(abstract)
	topic = set(topic) 

	return len(abstract & topic)/float(len(abstract | topic))

def parse(data):
	ans = {}
	for line in data:
		topic,values = line.split(':')
		values = dict([item.split('*')[::-1] for item in values.split(' + ')])
		ans[topic] = values
	return ans

#--load topics
topics = parse(open('topics','rb').read().splitlines())
all_words = list(set([item for sublist in topics.values() for item in sublist.keys()]))

#--load abstracts
abstracts = [line.split() for line in open('cleansed_abstracts','rb').read().splitlines()]

overlap = np.array([[jaccard_index(abstract,topics[topic].keys()) for abstract in abstracts] for topic in topics])
#-Each row is a topic, each column is an abstract

SAVENAME = 'jaccard_overlap'
if not os.path.isfile(SAVENAME):
	np.savetxt(SAVENAME,overlap,fmt='%.04f',delimiter='\t',header='Each row is a topic, each column an abstract')

idx = np.argsort(overlap,axis=1)
print idx
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(np.sort(overlap,axis=1),interpolation='nearest',aspect='auto')

plt.colorbar(cax)
plt.tight_layout()
plt.show()