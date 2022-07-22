'''
Data transformations:
    Filter data (torch-geometric Data -> Bool)
    Transform data (torch-geometric Data -> torch-geometric Data)
'''

import torch
from torch_geometric.transforms import BaseTransform

def max_filter(val,label=None):
    if label is None:
        return lambda graph: graph.y <= val
    return lambda graph: graph.y[label] <= val

def min_filter(val,label=None):
    if label is None:
        return lambda graph: graph.y >= val
    return lambda graph: graph.y[label] >= val

class LabelSelection(BaseTransform):
    def __init__(self,label):
        self.label = label

    def __call__(self,graph):
        if self.label == 'both':
            graph.y = torch.tensor([graph.y['h11'],graph.y['h21']],dtype=torch.int)
        else:
            graph.y = graph.y[self.label]
        return graph

class EdgeEnhancing(BaseTransform):
    def __call__(self,graph):
        new_features = torch.empty((0,10),dtype=torch.float)
        for i in range(graph.edge_index.shape[1]):
            node_features = torch.cat((graph.x[graph.edge_index[0,i],2:],
                                       graph.x[graph.edge_index[1,i],2:]))
            node_features = torch.unsqueeze(node_features,dim=0)
            new_features = torch.cat((new_features,node_features))
        edge_attr = torch.cat((graph.edge_attr,new_features),dim=1)
        graph.edge_attr = edge_attr
        return graph

def accuracy(prediction,target):
    return torch.mean(torch.eq(torch.round(prediction),torch.round(target)).type(torch.float))