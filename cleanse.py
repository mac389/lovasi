import twokenize as Tokenizer

from nltk.corpus import stopwords
from optparse import OptionParser
from progress.bar import Bar

'''
	Sources: some of this page is adapted from: [http://radimrehurek.com/gensim/tut1.html]
'''

#--Command  line parsing
op = OptionParser()
op.add_option('--i', dest='source', type='str', help='Filename of unprocessed text.')
op.add_option('--o', dest='destination', type='str', help='Filename for processed text.')
op.add_option('--stopwords', dest='stopwords_filename',type='str',help='Source of stopwords, default is local file called stopwords')
op.print_help()
#

opts,args = op.parse_args()
if len(args) > 0:
	op.error('This script only takes arguments preceded by command line options.')
	sys.exit(1)

custom_stopwords  = set(stopwords.words('english') if not opts.stopwords_filename else open('stopwords','rb').read().splitlines() + stopwords.words('english'))

text = open(opts.source,'rb').read().splitlines()

bar = Bar('Cleansing %s'%opts.source,max=len(text))
for i,item in enumerate(text):
	text[i] = [word for word in Tokenizer.tokenizeRawTweetText(item) if word not in custom_stopwords]
	# [ ] Use better tokenizer WE ALREADY HAVE A BETTER TOKENIZER
	bar.next()
bar.finish()

#Remove words that open appeal once
all_tokens = sum(text,[])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word)==1)
text = [[word for word in item if word not in tokens_once] for item in text]

with open(opts.destination,'wb') as f:
	for item in text:
		print>>f, ' '.join(item)