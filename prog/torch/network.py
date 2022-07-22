'''
Neural networks setup:
    Creation of torch neural networks
'''

import torch
from torch.nn import Module,BatchNorm1d,Dropout
from pytorch_lightning import LightningModule
from torch_geometric.nn import BatchNorm

class FullLayer(Module):
    def __init__(self,torch_layer,activation=None,batch_normalization=False,dropout=0):
        super().__init__()
        self.layer = torch_layer
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.dropout = dropout

    def forward(self,x):
        if isinstance(x,tuple):
            x = x[0]
        output = self.layer(x)
        if self.activation is not None:
            output = self.activation(output)
        if self.batch_normalization:
            channels_nb = output.shape[1]
            output = BatchNorm1d(channels_nb)(output)
        if self.dropout > 0:
            output = Dropout(self.dropout)(output)
        return output

class GraphLayer(Module):
    def __init__(self,geometric_layer,activation=None,batch_normalization=False,
                 dropout=0,inputs=['x','edge_index','edge_attr'],outputs=['x'],pooling=False):
        super().__init__()
        self.layer = geometric_layer
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.inputs = inputs
        self.outputs = outputs
        self.pooling = pooling

    def forward(self,batch):
        inputs_dict = {}
        for attr in self.inputs:
            inputs_dict[attr]=getattr(batch,attr)
        if len(inputs_dict) > 1:
            output = self.layer(**inputs_dict)
        else:
            output = self.layer(*inputs_dict.values())
        if self.activation is not None:
            output = self.activation(output)
        if self.batch_normalization:
            channels_nb = output.shape[1]
            output = BatchNorm(channels_nb)(output)
        if self.dropout > 0:
            output = Dropout(self.dropout)(output)
        if self.pooling:
            return output
        if len(self.outputs) > 1:
            raise ValueError('Several outputs not implemented')
        if self.outputs[0] == 'x':
            batch.x = output
        elif self.outputs[0] == 'edge_attr':
            batch.edge_attr = output
        return batch

class DebugLayer(Module):
    def forward(self,batch):
        print(batch)
        return batch

class LightningNetwork(LightningModule):
    def __init__(self,module,loss,metric,optimizer,optimizer_params,batch_size):
        super().__init__()
        self.module = module
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size

    def training_step(self,batch,batch_index):
        expected_output = torch.unsqueeze(torch.tensor(batch.y,dtype=torch.float),dim=1)
        output = self.module(batch)
        loss = self.loss(output,expected_output)
        metric = self.metric(output,expected_output)
        self.log('train_loss',loss,batch_size=self.batch_size)
        self.log('train_metric',metric,batch_size=self.batch_size)
        return loss

    def validation_step(self,batch,batch_index):
        expected_output = torch.unsqueeze(torch.tensor(batch.y,dtype=torch.float),dim=1)
        output = self.module(batch)
        loss = self.loss(output,expected_output)
        metric = self.metric(output,expected_output)
        self.log('train_loss',loss,batch_size=self.batch_size)
        self.log('train_metric',metric,batch_size=self.batch_size)

    def test_step(self,batch,batch_index):
        expected_output = torch.unsqueeze(torch.tensor(batch.y,dtype=torch.float),dim=1)
        output = self.module(batch)
        loss = self.loss(output,expected_output)
        metric = self.metric(output,expected_output)
        self.log('test_loss',loss,batch_size=self.batch_size)
        self.log('test_metric',metric,batch_size=self.batch_size)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(),**self.optimizer_params)
