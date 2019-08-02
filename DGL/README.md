# Goal
* Learn `DGL` from high level to calculate graphics.
* DGL is used to train a `simple graph neural network` to classify the nodes in the graph.
# Libraries
 * DGL 0.3
 * PyTorch 1.1.0
 * networkX 2.3+ (***A Visualization Tool for Displaying Graph Relations***)
# Problem Description
First, build two social networks as shown in the figure.
![]()
The task is to predict which party each member would like to join (0 or 33) based on the social network.
# code step
## Constructing Relational Graph Network
* Instantiating DGL classes
* Add 34 nodes into the graph; nodes are labeled from 0~33
* All 78 edges as a list of tuples
* Add edges two lists of nodes: src and dst
* Edges are directional in DGL; make them bi-directional
## Constructing Network Code of Diagram and Visualization Method
```python
import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
def build_karate_club_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                 (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
                 (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
                 (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                 (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
                 (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                 (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                 (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                 (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                 (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                 (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                 (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                 (33, 31), (33, 32)]

   # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list)) 
    print(src)
    #(1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7,8, 8, 9, 10, 10, 10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21, 25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33)
    print(dst)
    #(0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23, 24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32)
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional 
    g.add_edges(dst, src)
    return g


if __name__ == '__main__':

    G = build_karate_club_graph()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    #We have 34 nodes.
    #We have 156 edges.

    fig, ax = plt.subplots()
    fig.set_tight_layout(False)

    'import networkx as nx   import matplotlib.pyplot as plt 我们还可以通过将图转换为networkx图来可视化它：'
    #Since the actual graph is undirected, we convert it for visualization purpose.
    nx_G = G.to_networkx().to_undirected()
    # Position nodes using Kamada-Kawai path-length cost-function.  Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)
    #Draw the graph nx_G with Matplotlib.
    nx.draw(nx_G, pos, with_labels=True, node_color=[[0.7, 0.7, 0.7]]) #node_color 节点颜色的深浅程度
    plt.show()
```
[DGL](https://docs.dgl.ai/tutorials/models/index.html)
# Graph Convolutional Network (GCN)
```python
import torch
import torch.nn as nn

'定义 message 和 reduce 函数'
#注意：本教程忽略GCN标准化常量c_ij。
def gcn_message(edges):
    #参数是一批边。它使用源节点的特征'h'计算一个名为'msg'的（批处理）消息。
    """
    compute a batch of message called 'msg' using the source nodes' feature 'h'
    :param edges:
    :return:
    """
    return {'msg': edges.src['h']}

def gcn_reduce(nodes):
    #参数是一批节点。这通过将每个节点收到的'msg'相加来计算新的'h'特征。
    """
    compute the new 'h' features by summing received 'msg' in each node's mailbox.
    :param nodes:
    :return:
    """
    '??????'
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

'定义GCNLayer Module  （一层GCN） '
class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)


    def forward(self, g, inputs):
        # g—graph
        #inputs—input node features

        # first set the node features首先设置节点特征
        g.ndata['h'] = inputs
        # trigger message passing on all edges 在所有边上触发消息传送
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes 在所有节点上触发聚合
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features 获取结果节点特征
        h = g.ndata.pop('h')
        # perform linear transformation 执行线性变换
        return self.linear(h)

#通常，节点发送通过message函数计算的信息，并使用reduce函数聚合传入信息。
#然后，我们定义了一个包含两个GCN层的更深层次的GCN模型：

'定义一个2层的GCN 网络'
class GCN(nn.Module):

    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)    #两层
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h) #
        h = self.gcn2(g, h)

        return h


net = GCN(34, 5, 2)
print(net)
G = build_karate_club_graph()


'数据准备和初始化'
#我们使用one—hot向量来初始化节点特征。 由于这是半监督设置，因此仅向教练（节点0）和俱乐部主席（节点33）分配标签。 实施如下
inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])

'训练'
#训练循环与其他PyTorch模型完全相同。
# （1）创建一个优化器
# （2）将输入数据喂给模型
# （3）计算损失
# （4）使用autograd来优化模型。

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []

for epoch in range(30): #训练20次
    logits = net(G, inputs) #返回的h保存在logits中
   #为了之后的可视化保存logits
    'detach()?????'
    all_logits.append(logits.detach())
    #返回的是个概率？
    # 非线性函数torch.nn.functional.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
    logp = F.log_softmax(logits, 1)


    # 只计算标记节点的损失
    # logp-labels
    # torch.nn.functional.nll_loss(input, target, weight=None, size_average=True)
    # size_average (bool, optional) – 默认情况下，是mini-batchloss的平均值，然而，如果size_average=False，则是mini-batchloss的总和。
    loss = F.nll_loss(logp[labeled_nodes], labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss))
```
