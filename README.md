
```shell
git clone https://github.com/overlap-ai/words2map.git
cd words2map
./install.sh
source activate words2map
```

```python
from words2map import *
model = load_derived_vectors("passions.txt")
print k_nearest_neighbors(model=model, k=10, word="Data_Scientists")
```

<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/architecture.png" /></span></p>

<p style="text-align: center;"><span style="font-family:georgia,serif"><img alt="" src="https://raw.githubusercontent.com/overlap-ai/words2map/master/visualizations/tech.png" /></span></p>
