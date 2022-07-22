'''
Network testing:
    Define dataset
    Define network
    Train and test network
'''

import shutil
from torch.utils.data import random_split
from dataset import OriginalCICY3,FavorableCICY3

from transformations import min_filter,max_filter,LabelSelection,EdgeEnhancing,accuracy
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.data import LightningDataset

from network import LightningNetwork,GraphLayer,FullLayer
from torch_geometric.nn import GINEConv
from addition import GlobalPooling
from torch.nn import Sequential,Linear,ReLU
import torch.nn.functional as F
from torch.optim import Adam
from pytorch_lightning import Trainer

shutil.rmtree('data/processed',ignore_errors=True)

FAVORABLE = False
TARGET ='h21'
BATCH_SIZE = 32
EPOCHS = 5000
NODE_ENHANCING = True
EDGE_ENHANCING = False
NODE_ENCODING = False
EDGE_ENCODING = False
ENCODING_SIZE = 32

node_size,edge_size = 2,1
if TARGET == 'h11':
    MIN,MAX = 1,16
else:
    MIN,MAX = 1,91
filters = [min_filter(MIN,TARGET),max_filter(MAX,TARGET)]
pre_transforms = [LabelSelection(TARGET)]
if NODE_ENHANCING:
    pre_transforms.append(LocalDegreeProfile())
    node_size += 5
if EDGE_ENHANCING:
    pre_transforms.append(EdgeEnhancing())
    edge_size += 2*(node_size-2)
if FAVORABLE:
    dataset = FavorableCICY3(root='data',pre_filter=filters,pre_transform=pre_transforms)
else:
    dataset = OriginalCICY3(root='data',pre_filter=filters,pre_transform=pre_transforms)
train_dataset, val_dataset, test_dataset = random_split(dataset,[0.8,0.1,0.1])
datamodule = LightningDataset(train_dataset,val_dataset,test_dataset,batch_size=BATCH_SIZE)

layers_list = []
if NODE_ENCODING:
    node_encoder = Sequential(Linear(node_size,32),ReLU(),Linear(32,ENCODING_SIZE))
    layers_list.append(GraphLayer(node_encoder,activation=ReLU(),batch_normalization=True,
                                  inputs=['x'],outputs=['x']))
    node_size = ENCODING_SIZE
if EDGE_ENCODING:
    edge_encoder = Sequential(Linear(edge_size,32),ReLU(),Linear(32,ENCODING_SIZE))
    layers_list.append(GraphLayer(edge_encoder,activation=ReLU(),batch_normalization=True,
                                  inputs=['edge_attr'],outputs=['edge_attr']))
    edge_size = ENCODING_SIZE

layers_list += [
    GraphLayer(GINEConv(Sequential(Linear(node_size,128),ReLU(),Linear(128,64)),edge_dim=edge_size),
               activation=ReLU(),batch_normalization=True),
    GraphLayer(GINEConv(Sequential(Linear(64,128),ReLU(),Linear(128,256)),edge_dim=edge_size),
               activation=ReLU(),batch_normalization=True),
    GraphLayer(GlobalPooling(aggr='add'),
               inputs=['x','batch'],pooling=True),
    #FullLayer(Linear(128,256),activation=ReLU(),batch_normalization=True),
    FullLayer(Linear(256,128),activation=ReLU(),batch_normalization=True),
    FullLayer(Linear(128,64),activation=ReLU(),batch_normalization=True),
    FullLayer(Linear(64,1),activation=ReLU())
]

network = Sequential(*layers_list)

optimizer_params = {'lr':0.001}
loss = F.mse_loss
metric = accuracy
lightning_network = LightningNetwork(module=network,loss=loss,metric=metric,optimizer=Adam,
                                     optimizer_params=optimizer_params,batch_size=BATCH_SIZE)
trainer = Trainer(max_epochs=EPOCHS)
if __name__ == '__main__':
    trainer.fit(lightning_network,datamodule)
    trainer.test(lightning_network,datamodule)
