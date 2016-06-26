import semidbm, cPickle, os, time
import numpy as np
from gensim.models import Word2Vec

def load_model():
	directory = os.getcwd() + "/vectors"
	model = cPickle.load(open(os.path.join(directory, 'model.pickle')))
	model.vocab = DataLoader(os.path.join(directory, 'word_to_index'))
	model.index2word = DataLoader(os.path.join(directory, 'index_to_word'))
	model.syn0norm = np.memmap(os.path.join(directory, 'syn0norm.dat'), dtype='float16', mode='r', shape=(len(model.vocab.keys()), model.layer1_size))
	model.syn0 = model.syn0norm
	return model

class DataLoader(dict):
	def __init__(self, dbm_file):
		self._dbm = semidbm.open(dbm_file, 'r')
	def __iter__(self):
		return iter(self._dbm.keys())
	def __len__(self):
		return len(self._dbm)
	def __contains__(self, key):
		if isinstance(key, int):
			key = str(key)
		return key in self._dbm
	def __getitem__(self, key):
		if isinstance(key, int):
			key = str(key)
			return self._dbm[key]
		else:
			return cPickle.loads(self._dbm[key])
	def keys(self):
		return self._dbm.keys()
	def values(self):
		return [self._dbm[key] for key in self._dbm.keys()]
	def itervalues(self):
		return (self._dbm[key] for key in self._dbm.keys())

print "Loading 100k vectors of 300 dimensions each..."
model = load_model()
print "The vector for \'intelligence\':\n{}".format(model["intelligence"])