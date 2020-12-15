<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/architecture.png" /></span></p>
 
> *This project is deprecated and no longer supported. The basic ideas are still useful:*  
> *(1) derive new vectors at runtime by searching related websites online, combining words found with known vectors*  
> *(2) embed high dimensional derived vectors into a lower dimensional space for visualization*  
> *For the 2nd objective, when words2map was originally developed t-SNE made the most sense for dimensionality reduction.*  
> *At the time of this writing however (2020), I would recoomend using [UMAP](https://github.com/lmcinnes/umap).*  
> *See this [archived blog post](http://web.archive.org/web/20160806040004if_/http://blog.yhat.com/posts/words2map.html) for more details on the algorithms*


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
git clone https://github.com/overlap-ai/words2map.git
cd words2map
./install.sh
```

<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/passions.png" /></span></p>
