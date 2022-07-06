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
