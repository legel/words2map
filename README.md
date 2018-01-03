<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/architecture.png" /></span></p>

```shell
# installation
git clone https://github.com/overlap-ai/words2map.git
cd words2map
./install.sh
source activate words2map
```

```python
from words2map import *

# derive vectors for unknown words the same way humans do: google
model = load_model()
words = load_words("passions.csv")
vectors = [derive_vector(word, model) for word in words]
save_derived_vectors(words, vectors, "passions.txt")

# analyze nearest neighbors of derived word vectors 
model = load_derived_vectors("passions.txt")
print k_nearest_neighbors(model=model, k=10, word="Data_Scientists")

# visualize clusters of word vectors in 2D
words = [word for word in model.vocab]
vectors = [model[word] for word in words]
vectors_in_2D = reduce_dimensionality(vectors)
generate_clusters(words, vectors_in_2D)
```
<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/tech.png" /></span></p>
