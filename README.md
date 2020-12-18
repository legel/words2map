<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/architecture.png" /></span></p>
 
> *This project is deprecated and no longer supported. The core novel contribution could still useful to modern NLP R&D:*  
> *Instantly derive [OOV](https://medium.com/@shabeelkandi/handling-out-of-vocabulary-words-in-natural-language-processing-based-on-context-4bbba16214d5) vectors by searching online and combining vectors of known words.  Hooking a vector-based NLP system into a search engine like Google could still work. The basic idea is search out-of-vocabulary words and phrases as soon as they're encountered - just like a human would - and then parse the top N websites that are related to these.  For this project, N = 50.  Through a variety of classical NLP techniques for prioritizing the most important distinctive words on every website, which alredy exist in a vector corpus, all of these vectors are the summed to derive a new word vector.  Here is how that summation looks:*  

<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/legel/words2map/master/visualizations/human_robot_cyborg.png" /></span></p>


> *In practice the derived vectors from autonomous real-time online research tend to be surprisingly good.  All of the following vectors were derived in this way:*  
<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/passions.png" /></span></p>

> *As another quick update, the above visualization used t-SNE for reducing dimensionality from 300D to 2D, and then HDBSCAN for clustering.*  
> *I would still recommend the HDBSCAN algorithm for any topologically sensitive clustering application, but for dimensionality reduction I would now recommend [UMAP](https://github.com/lmcinnes/umap).*  

> *See this [archived blog post](http://web.archive.org/web/20160806040004if_/http://blog.yhat.com/posts/words2map.html) for more details on the words2map algorithm.*

### Documentation: Derive new vectors for words by searching online

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

