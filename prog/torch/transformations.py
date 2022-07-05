'''
Data transformations:
    Filter data (torch-geometric Data -> Bool)
    Transform data (torch-geometric Data -> torch-geometric Data)
'''

import torch
from torch_geometric.utils import degree

def max_filter(val,label=None):
    if label is None:
        return lambda graph: graph.y <= val
    return lambda graph: graph.y[label] <= val

def min_filter(val,label=None):
    if label is None:
        return lambda graph: graph.y >= val
    return lambda graph: graph.y[label] >= val

def label_selection(label):
    def select(graph):
        if label == 'both':
            graph.y = torch.tensor([graph.y['h11'],graph.y['h21']],dtype=torch.int)
        else:
            graph.y = graph.y[label]
        return graph
    return select

def node_degree(graph):
    nodes_degrees = degree(graph)

