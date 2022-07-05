'''
Network testing:
    Define dataset
    Define network
    Train and test network
'''

from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from dataset import OriginalCICY3
from network import LightningNetwork
from transformations import min_filter,max_filter,label_selection

filters = [min_filter(1,'h21'),max_filter(91,'h21')]
pre_transforms = [label_selection('h21')]

dataset = OriginalCICY3(root='data',pre_filter=filters,pre_transform=pre_transforms)

train_dataset, val_dataset, test_dataset = random_split(dataset,[0.8,0.1,0.1])
graph = train_dataset[0]
print(graph.x,graph.edge_index,graph.edge_attr)
