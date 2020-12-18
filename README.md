<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/architecture.png" /></span></p>
 
> *Deriving [OOV](https://medium.com/@shabeelkandi/handling-out-of-vocabulary-words-in-natural-language-processing-based-on-context-4bbba16214d5) deep vectors by searching online.*  
> *(1) Connect NLP vector database with a web search engine API like Google / Bing*  
> *(2) Do a web search on out-of-vocabulary words (just like a human would)*  
> *(3) Parse N-grams (e.g. N = 5) for all text from top M websites (e.g. M = 50)*  
> *(4) Filter known N-grams on large pre-trained corpus (e.g. word2vec, with 3 million N-grams)*  
> *(5) Rank N-grams: inverse global frequency in vocab x local frequency on M websites (i.e. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf))*  
> *(6) Sum the top V (e.g. V = 25) N-grams, element-wise, to derive the new vector, i.e.*   

<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/legel/words2map/master/visualizations/human_robot_cyborg.png" /></span></p>

> *(7) Visualize quality by reducing dimensions to 2D or 3D (e.g [t-SNE](https://lvdmaaten.github.io/tsne/) works, but [UMAP](https://github.com/lmcinnes/umap) now recommended)*  
> *(8) Finally, show clusters with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan), color-coded in a perceptually uniform space*  

> *These "words2map" were all derived as explained above:*  
<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/passions.png" /></span></p>
<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/famous.png" /></span></p>

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

