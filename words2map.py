# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from hdbscan import HDBSCAN
from nltk import tokenize, bigrams, trigrams, everygrams, FreqDist, corpus
from pattern.web import Google, SEARCH, download, plaintext, HTTPError, HTTP403Forbidden, URLError, URLTimeout, SearchEngineLimitError
from ssl import SSLError
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import semidbm
from cPickle import loads, load, UnpicklingError
from operator import itemgetter
from itertools import combinations
from unidecode import unidecode
import string
import time
import sys
import os
from os import listdir, getcwd
from os.path import isfile, join
from multiprocessing import Process, Manager

GOOGLE_API_KEY = "AIzaSyB4f-UO51_qDWXIwSwR92aejZso6hHJEY4" # Google provides everyone with 100 free searches per day, and then $5 per 1,000 searches after that, and a limit of 10,000 per day. However new users get $300 free in their first 60 days, so that's your first 60,000 words2map free.

class Loader(dict):
	# loads ~20 MB of indexes for word2vec and index2word into RAM for fast dictionary reads
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
			return loads(self._dbm[key])
	def keys(self):
		return self._dbm.keys()
	def values(self):
		return [self._dbm[key] for key in self._dbm.keys()]
	def itervalues(self):
		return (self._dbm[key] for key in self._dbm.keys())

def load_model():
	# current model contains 100,000 vectors of 300 elements, each element containing a 16 bit floating point number, such that vectors total ~60 MB uncompressed; note that there is practically no loss of precision in saving vectors with 16 bits versus 32 bits, while data consumption is halved...
	print "Loading 100,000 word vectors..."
	directory = getcwd() + "/vectors"
	model = load(open(join(directory, 'model.pickle')))
	model.vocab = Loader(join(directory, 'word_to_index'))
	model.index2word = Loader(join(directory, 'index_to_word'))
	model.syn0norm = np.memmap(join(directory, 'syn0norm.dat'), dtype='float16', mode='r', shape=(len(model.vocab.keys()), model.layer1_size))
	model.syn0 = model.syn0norm
	return model

def get_collocations(words):
	# returns n-grams up to trigrams that appear at least 3 times, with pruning of grams that are redundant
	minimum_frequency = 3
	ngrams = {"_".join(ngram): frequency/len(words) for ngram, frequency in FreqDist(everygrams(words, max_len=3)).items() if frequency > minimum_frequency}
	collocations = dict(ngrams)
	for ngram, likelihood in dict(ngrams).iteritems():
		grams = ngram.split("_")
		if len(grams) != 1:
			gram_likelihoods = [ngrams[gram] for gram in grams]
			if likelihood < 0.5 * np.prod(gram_likelihoods)**(1 / len(grams)):
				collocations.pop(ngram, None)
			else:
				for gram in grams:
					collocations.pop(gram, None)
	return sorted(collocations.items(), key=itemgetter(1), reverse=True)

def evaluate_keyword(frequency, word_index, max_word_index=100000):
	# inspired by tf-idf (https://en.wikipedia.org/wiki/tf–idf)
	rarity = word_index / max_word_index # intuition: rare words tend to have bigger indexes in word2vec, because they're more likely encountered later in training
	return frequency * rarity

def extract_keywords(url, model, all_keywords):
	minimum_word_index_for_unigrams = 20000
	try:
		text = plaintext(download(url))
		words = [word for word in tokenize.word_tokenize(text) if word.isalnum() and word.lower() not in corpus.stopwords.words('english') and word.lower() not in url]
		for collocation, frequency in get_collocations(words):
			word_index = get_index(collocation, model)
			if word_index:
				if collocation.count('_') == 0 and word_index < minimum_word_index_for_unigrams:
					pass
				else:
					all_keywords[collocation] = all_keywords.get(collocation, 0) + evaluate_keyword(frequency, word_index)
	except (URLError, URLTimeout, HTTPError, HTTP403Forbidden, SSLError, UnicodeEncodeError, ValueError) as e:
		pass
		
def research_keywords(something_unknown, model, websites_to_scan=100, keyword_count=50, attempts=0):
	# searches for something unknown on Google to find related websites and returns a ranked list of keywords from across all sites
	maximum_number_of_google_search_attempts = 3
	if attempts < maximum_number_of_google_search_attempts:
		all_keywords = Manager().dict()
		engine = Google(license=GOOGLE_API_KEY, throttle=1.0, language="en")
		try:
			processes = []
			for page in range(int(websites_to_scan/10)):
				for website in engine.search(something_unknown, start=page+1, count=10, type=SEARCH, cached=False):
					web_mining_process = Process(target=extract_keywords, args=(website.url, model, all_keywords))
					processes.append(web_mining_process)
					web_mining_process.start()
				[process.join() for process in processes]
		except HTTP403Forbidden:
			print "\nToday's maximum number of free searches from Google shared by this API across all words2map users has expired.\nPlease get your own key at https://code.google.com/apis/console\n\nFrom that site, simply:\n1. In the API Manager Overview, find \"Custom Search API\" and enable it\n2. Copy your new API key from \"Credentials\"\n3. Paste it in words2map.py in the global variable \"GOOGLE_API_KEY\"\n"
			sys.exit(1)
		except (URLError, URLTimeout, HTTPError, SSLError):
			print "\nUnable to reach Google Search for {}, trying one more time".format(something_unknown)
			return research_keywords(something_unknown, model, attempts=attempts+1)
		all_keywords = sorted(all_keywords.items(), key=itemgetter(1), reverse=True)
		print "\nKeywords about {} to combine vectors for:".format(something_unknown)
		top_keywords = []
		for i in range(keyword_count):
			try:
				keyword, score = all_keywords[i]
				top_keywords.append(all_keywords[i])
				print "{} {}".format(round(score, 3), unidecode(keyword))
			except IndexError:
				break
		return top_keywords
	else:
		print "After a few tries, it seems that Google is not returning results for us. If you haven't done so already, please try adding your own API key at https://code.google.com/apis/console\n\nFrom that site, simply:\n1. In the API Manager Overview, find \"Custom Search API\" and enable it\n2. Copy your new API key from \"Credentials\"\n3. Paste it in words2map.py in the global variable \"GOOGLE_API_KEY\"\n"
		sys.exit(1)

def load_derived_vectors(filename):
	# loads derived vectors from a previous words2map as a standalone Gensim Word2Vec model (https://radimrehurek.com/gensim/models/word2vec.html)
	filepath = getcwd() + "/derived_vectors/" + filename
	model = Word2Vec.load_word2vec_format(filepath, binary=False)
	return model

def save_derived_vectors(words, vectors):
	# saves vectors in human readable format, which can be easily and quickly reloaded back into a Gensim Word2Vec model (https://radimrehurek.com/gensim/models/word2vec.html)
	derived_vectors_directory = getcwd() + "/derived_vectors"
	files = [f for f in listdir(derived_vectors_directory) if isfile(join(derived_vectors_directory, f))]
	words2map_files = [int(f.split("_")[1].split(".txt")[0]) for f in files if "words2map_" in f and ".txt" in f]
	if words2map_files:
		map_number = max(words2map_files) + 1
	else:
		map_number = 0
	filename = "words2map_{}.txt".format(map_number)
	f = open("{}/{}".format(derived_vectors_directory, filename),'w')
	f.write("{} {}\n".format(len(words), 300)) 
	for word, vector in zip(words, vectors):
		formatted_word = word.replace(" ", "_")
		formatted_vector = ' '.join([str(i) for i in vector])
		f.write("{} {}\n".format(formatted_word, formatted_vector))
	f.close()
	print "Saved word vectors as {}".format(filename)
	return filename

def test_performance():
	# calculates average time to access a word vector after loading the model in RAM 
	model = load_model()
	times = []
	for i in range(100000):
		word = model.index2word[i]
		start_time = time.time()
		vector = model[word]
		end_time = time.time()
		times.append(end_time - start_time)
	total_time = sum(times)
	average_time = np.mean(times)
	print "You can count on it taking about {} μs to check / get each word vector at runtime, after loading the model".format(round(total_time, 2), round(average_time * 100000, 2))

def visualize_as_clusters(words, vectors_in_2D):
	# HDBSCAN, i.e. hierarchical density-based spatial clustering of applications with noise (https://github.com/lmcinnes/hdbscan)
	vectors = vectors_in_2D
	sns.set_context('poster')
	sns.set_style('white')
	sns.set_color_codes()
	plot_kwds = {'alpha' : 0.5, 's' : 400, 'linewidths': 0}
	labels = HDBSCAN(min_cluster_size=2).fit_predict(vectors)
	palette = sns.color_palette("husl", np.unique(labels).max() + 1)
	colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
	plt.figure(figsize=(25, 25))
	plt.scatter(vectors.T[0], vectors.T[1], c=colors, **plot_kwds)
	plt.axis('off')
	x_vals = [i[0] for i in vectors]
	y_vals = [i[1] for i in vectors]
	for i, word in enumerate(words):
		word = unidecode(word).replace("_", " ")
		plt.annotate(word, (x_vals[i], y_vals[i]))
	visualizations = getcwd() + "/visualizations"
	files = [f for f in listdir(visualizations) if isfile(join(visualizations, f))]
	words2map_files = [int(f.split("_")[1].split(".png")[0]) for f in files if "words2map_" in f and ".png" in f]
	if words2map_files:
		map_number = max(words2map_files) + 1
	else:
		map_number = 0
	print "\nVisualization saved! Check out words2map_{}.png".format(map_number)
	plt.savefig("{}/words2map_{}.png".format(visualizations, map_number))

def reduce_dimensionality(vectors, dimensions=2):
	# t-stochastic neighbor embedding (https://lvdmaaten.github.io/tsne/)
	print "\nComputing t-SNE reduction of 300D word vectors to {}D".format(dimensions)
	tsne_model = TSNE(n_components=dimensions, n_iter=250000000, metric="correlation", learning_rate=50, early_exaggeration=500.0, perplexity=30.0)
  	np.set_printoptions(suppress=True)
	vectors_in_2D = tsne_model.fit_transform(np.asarray(vectors).astype('float64'))
	return vectors_in_2D

def k_nearest_neighbors(model, k=10, word=None, vector=None):
	if word:
		return model.most_similar(positive=[word], topn=k)
	elif any(vector):
		return model.most_similar(positive=[vector], topn=k)
	else: 
		raise ValueError("Provide a word or vector as an argument to get k-nearest neighbors\ne.g. k_nearest_neighbors(k=25, word=\"humanity\")")

def get_vector(word, model, lowercase=True):
	# returns vector of word as 300 dimensions, each containing a 16 bit floating point number, or None if word doesn't exist
	if lowercase:
		formatted_word = word.replace(" ", "_").lower()
		try:
			vector = model[formatted_word] 
			return np.asarray(vector)
		except (EOFError, KeyError, UnpicklingError):
			return get_vector(word, model, lowercase=False)
	else:
		formatted_word = word.replace(" ", "_")
		try:
			vector = model[formatted_word] 
			return np.asarray(vector)
		except (EOFError, KeyError, UnpicklingError):
			return None



def get_index(word, model, lowercase=True):
	# returns index of word ranging between 0 and 99,999 (corresponding to the order that words were encountered during word2vec training), trying first for the lowercase version if that's set to True, and returning None if neither exists
	if lowercase:
		formatted_word = word.replace(" ", "_").lower()
		try:
			word_index = model.vocab[formatted_word].index
			return word_index
		except (EOFError, KeyError, UnpicklingError):
			return get_index(word, model, lowercase=False)
	else:
		formatted_word = word.replace(" ", "_")
		try:
			word_index = model.vocab[formatted_word].index
			return word_index
		except (EOFError, KeyError, UnpicklingError):
			return None

def add_vectors(vectors):
	# vector addition is done first by averaging the values for each dimension, and then unit normalizing the derived vector (e.g. https://youtu.be/BD8wPsr_DAI)
	derived_vector = np.average(np.array(vectors), axis=0)
	return derived_vector / np.linalg.norm(derived_vector)

def derive_vector(word, model):
	# extracts keywords from Google searches and adds their vectors
	keywords = research_keywords(word, model)
	vectors = [get_vector(keyword, model) for keyword, score in keywords]
	return add_vectors(vectors)

def clarify(words):
	#returns vectors for any set of words, and visualizes these words in a 2D plot
	model = load_model()
	vectors = [derive_vector(word, model) for word in words]
	# filename = save_derived_vectors(words, vectors)
	# model = load_derived_vectors("words2map_0.txt")
	# words = [word for word in model.vocab]
	# vectors = [model[word] for word in words]
	# vectors_in_2D = reduce_dimensionality(vectors)
	# visualize_as_clusters(words, vectors_in_2D)

if __name__ == "__main__":
	words = ["Yann LeCun"]
	#words = ["Larry Page", "Elon Musk", "Sebastian Thrun", "Andrew Ng", "Yoshua Bengio", "Yann LeCun", "Geoffrey Hinton", "Jürgen Schmidhuber", "Bruno Olshausen", "J.J. Hopfield", "Randall O\'Reilly", "Demis Hassabis", "Peter Norvig", "Jeff Dean", "Daphne Koller", "Gunnar Carlson", "Nate Silver", "Alex Pentland", "Hilary Mason", "Julia Hirschberg", "Chris Wiggins", "David Blei", "Michael I. Jordan", "Rocco Servedio", "Leslie Valiant", "Vladimir Vapnik", "Alan Turing", "Georg Cantor", "Alan Kay", "Thomas Bayes", "Ludwig Boltzmann", "Peter Dirichlet", "Carl Gauss", "Donald Knuth", "Claude Shannon", "Marvin Minsky", "John von Neumann", "Thomas J. Watson", "Ken Thompson", "Linus Torvalds", "Douglas Engelbart", "Grace Hopper", "Marissa Mayer", "Bill Gates", "Steve Jobs", "Steve Wozniak", "Jeff Bezos", "Mark Zuckerberg", "Eric Schmidt", "Sergey Brin", "Tim Berners Lee", "Stephen Wolfram", "Bill Joy", "Vint Cerf", "Paul Graham", "Richard Hamming", "Eric Horvitz", "Stephen Omohundro", "Jaron Lanier", "Bruce Schneier", "Ray Kurzweil", "Richard Socher", "Alex Krizhevsky", "Rajat Raina", "Adam Coates", "Léon Bottou", "Greg Corrado", "Marc'Aurelio Ranzato", "Honglak Lee", "Quoc V. Le", "Radim Řehůřek", "Tom De Smedt", "Chris Moody", "Christopher Olah", "Tomas Mikolov"]
	clarify(words)
