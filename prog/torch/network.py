'''
Neural networks setup:
    Creation of torch neural networks
'''
from torch.nn import Module,Sequential,SyncBatchNorm,Dropout
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
        output = self.layer(x)
        if self.activation is not None:
            output = self.activation(output)
        if self.batch_normalization:
            channels_nb = output.shape[1]
            output = SyncBatchNorm(channels_nb)(output)
        if self.dropout > 0:
            output = Dropout(self.dropout)(output)
        return output

class GraphLayer(Module):
    def __init__(self,geometric_layer,activation=None,batch_normalization=False,
                 dropout=0,need_index=False):
        super().__init__()
        self.layer = geometric_layer
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.need_index = need_index

    def forward(self,x):
        batch,index = x
        if self.need_index:
            output = self.layer(batch,index)
        else:
            output = self.layer(batch)
        if self.activation is not None:
            output = self.activation(output)
        if self.batch_normalization:
            channels_nb = output.shape[1]
            output = BatchNorm(channels_nb)(output)
        if self.dropout > 0:
            output = Dropout(self.dropout)(output)
        return output,index

class SequentialNeuralNetwork(Sequential):
    def __init__(self,layers):
        super().__init__()
        self.layers = layers

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class LightningNetwork(LightningModule):
    def __init__(self,module,loss,metric,optimizer):
        super().__init__()
        self.module = module
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer

    def training_step(self,batch,batch_idx):
        x,expected_output = batch
        output = self.module(x)
        loss = self.loss(output,expected_output)
        metric = self.metric(output,expected_output)
        self.log('train_loss',loss)
        self.log('train_metric',metric)
        return loss

    def test_step(self,batch,batch_idx):
        x,expected_output = batch
        output = self.module(x)
        loss = self.loss(output,expected_output)
        self.log('test_loss',loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
