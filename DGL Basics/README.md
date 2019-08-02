# Goal
* To create a graph.
* To read and write node and edge representations.
# Graph Creation
* The design of DGLGraph was influenced by other graph libraries. Indeed, you can create a graph from `networkx`, and convert it into a DGLGraph and vice versa:
```python
import networkx as nx
import dgl

g_nx = nx.petersen_graph()
g_dgl = dgl.DGLGraph(g_nx)

import matplotlib.pyplot as plt
plt.subplot(121)
nx.draw(g_nx, with_labels=True)
plt.subplot(122)
nx.draw(g_dgl.to_networkx(), with_labels=True)

plt.show()
```
They are the same graph, except that `DGLGraph` is always ***directional***.
***One can also create a graph by calling DGL’s own interface.***
Now let’s build a star graph. `DGLGraph nodes` are consecutive range of integers between `0 and number_of_nodes()` and can grow by calling add_nodes. DGLGraph edges are in order of their additions. Note that edges are accessed in much the same way as nodes, with one extra feature of edge broadcasting:
```python
import dgl
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
g = dgl.DGLGraph()
src = th.tensor(list(range(1, 10)))
g.add_edges(src, 0)
nx.draw(g.to_networkx(), with_labels=True)
plt.show()
```
![](https://github.com/ranran4082391/Weakly-Supervised-Model/blob/master/DGL%20Basics/dgl.png)
