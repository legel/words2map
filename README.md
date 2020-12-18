<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/architecture.png" /></span></p>
 
> *Instantly derive [OOV](https://medium.com/@shabeelkandi/handling-out-of-vocabulary-words-in-natural-language-processing-based-on-context-4bbba16214d5) vectors by searching online.  
> *Here is the algorithm:
> *(1) Hook up a vector-based NLP system with real-time OOV parsing requirements into a search engine API like Google / Bing*  
> *(2) When out-of-vocabulary (OOV) words and phrases are encountered, automatically search for them on the web - just like a human would*  
> *(3) Download and parse N-grams (e.g. N = 5) for all text from the top M websites (e.g. M = 50)
> *(4) Filter all N-grams that are already known in an existing vector vocublary (e.g. word2vec corpus from Google in 2013 with 3 million N-grams) 
> *(5) Weight and rank the important of all found N-grams by multiplying a metric of their global frequency in the vocabulary corpus, by a metric of local frequency (e.g. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)*  
> *(6) Sum the top T ranked vectors, with or without weighting.  In practice a simple element-wise sum without weighting can work well, i.e.*   

<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/legel/words2map/master/visualizations/human_robot_cyborg.png" /></span></p>

> *(7) Visualize quality of derivation by reducing dimensionality of all vectors to 2D / 3D (e.g [t-SNE](https://lvdmaaten.github.io/tsne/) was originally uesd, but now [UMAP](https://github.com/lmcinnes/umap) is recommended).
> *(8) Finally, show clusters with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) and color-code them in a perceptually uniformly distributed space

> *In practice the derived vectors from autonomous real-time online research tend to be surprisingly good.  All of the following vectors were derived in this way:*  
<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/passions.png" /></span></p>
<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/famous.png" /></span></p>
<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/tech.png" /></span></p>

> *See this [archived blog post](http://web.archive.org/web/20160806040004if_/http://blog.yhat.com/posts/words2map.html) for more details on the words2map algorithm.*

### Derive new vectors for words by searching online

```python
from words2map import *
model = load_model()
words = load_words("passions.csv")
vectors = [derive_vector(word, model) for word in words]
save_derived_vectors(words, vectors, "passions.txt")
```

### Analyze derived word vectors
```python
from words2map import *
from pprint import pprint
model = load_derived_vectors("passions.txt")
pprint(k_nearest_neighbors(model=model, k=10, word="Data_Scientists"))
```

### Visualize clusters of vectors
```python
from words2map import *
model = load_derived_vectors("passions.txt")
words = [word for word in model.vocab]
vectors = [model[word] for word in words]
vectors_in_2D = reduce_dimensionality(vectors)
generate_clusters(words, vectors_in_2D)
```

### Install 

```shell
# known broken dependencies: automatic conda installation, python 2 -> 3, gensim
# feel free to debug and make a pull request if desired
git clone https://github.com/overlap-ai/words2map.git
cd words2map
./install.sh
```

