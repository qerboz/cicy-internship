'''
Data creation:
    Turn cicy 3folds matrices into graphs
    Create torch-geometric datasets
'''
import os
import torch
import torch_geometric as pyg
import pandas as pd

def mat_to_graph(matrices):
    if isinstance(matrices,torch.Tensor):
        edges_list = []
        edges_features = []
        rows = matrices.shape[0]
        columns = matrices.shape[1]
        nodes_features = [[0,1]]*rows + [[1,0]]*columns
        nodes_nb = len(nodes_features)
        nodes_features = torch.tensor(nodes_features,dtype=torch.float)
        for i in range(rows):
            for j in range(columns):
                if matrices[i][j] > 0:
                    edges_list += [[i,rows+j],[rows+j,i]]
                    edges_features += [[matrices[i][j]]]*2
        edges_nb = len(edges_list)/2
        edges_list = torch.transpose(torch.tensor(edges_list,dtype=torch.int64),0,1)
        edges_features = torch.tensor(edges_features,dtype=torch.float)
        sparse_adj = torch.sparse_coo_tensor(edges_list,torch.tensor([1]*len(edges_features)),
                                             size=(nodes_nb,nodes_nb))
        mod_sparse_adj = torch.sparse_coo_tensor(edges_list,torch.squeeze(edges_features),
                                                 size=(nodes_nb,nodes_nb))
        return {'nodes':nodes_features,'edges_list':edges_list,
                'edges_features':edges_features,'adj_mat':sparse_adj,
                'mod_adj_mat':mod_sparse_adj}
    graph_list = []
    for mat in matrices:
        graph_list.append(mat_to_graph(mat))
    return graph_list

def mat_to_graph2(matrices):
    if isinstance(matrices,torch.Tensor):
        nodes_features = []
        edges_list = []
        edges_features = []
        rows = matrices.shape[0]
        columns = matrices.shape[1]
        node_num_mat = torch.zeros(matrices.shape,dtype=torch.float)
        node_num = 0
        for i in range(rows):
            for j in range(columns):
                if matrices[i][j] > 0:
                    node_num_mat[i,j] = node_num
                    nodes_features.append([matrices[i][j]])
                    for k in range(0,i):
                        if matrices[k][j] > 0:
                            edges_list.append([node_num_mat[k,j],node_num])
                            edges_features.append([1,0])
                            edges_list.append([node_num,node_num_mat[k,j]])
                            edges_features.append([1,0])
                    for k in range(0,j):
                        if matrices[i][k] > 0:
                            edges_list.append([node_num_mat[i,k],node_num])
                            edges_features.append([0,1])
                            edges_list.append([node_num,node_num_mat[i,k]])
                            edges_features.append([0,1])
                    node_num += 1
        edges_list = torch.transpose(torch.tensor(edges_list,dtype=torch.int),0,1)
        edges_features = torch.tensor(edges_features,dtype=torch.float)
        sparse_adj = torch.sparse_coo_tensor(edges_list,torch.tensor([1]*len(edges_features)),
                                             size=(len(nodes_features),len(nodes_features)))
        return {'nodes':nodes_features,'edges_list':edges_list,
                'edges_features':edges_features,'adj_mat':sparse_adj}
    graph_list = []
    for mat in matrices:
        graph_list.append(mat_to_graph2(mat))
    return graph_list

def graph_to_mat(graph):
    rows = 0
    columns = 0
    for node in graph['nodes']:
        if node == [0,1]:
            rows += 1
        else:
            columns += 1
    matrice = torch.zeros((rows,columns))
    for i in range(len(graph['edges_list'])):
        pos = graph['edges_list'][i]
        matrice[pos[0],pos[1]-rows] = graph['edges_features'][i]
    return matrice

def check_graph_mat(graph,mat):
    return (graph_to_mat(graph)==mat).all()

def dataframe_to_list(dataframe):
    graphs_list = mat_to_graph(list(dataframe['matrix'].values))
    for i,_ in enumerate(graphs_list):
        graphs_list[i]['target'] = {'h11':dataframe['h11'].values[i],
                                    'h21':dataframe['h21'].values[i]}
    return graphs_list

def graph_to_data(graph):
    return pyg.data.Data(x=graph['nodes'],edge_index=graph['edges_list'],
                         edge_attr=graph['edges_features'],y=graph['target'])

class OriginalCICY3(pyg.data.InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.url = 'http://www.lpthe.jussieu.fr/~erbin/files/data/cicy3o.h5'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def pre_filter_func(self,graph):
        if isinstance(self.pre_filter,list):
            for check in self.pre_filter:
                if not check(graph):
                    return False
            return True
        return self.pre_filter(graph)

    def pre_transform_func(self,graph):
        if isinstance(self.pre_transform,list):
            transform = pyg.transforms.Compose(self.pre_transform)
            return transform(graph)
        return self.pre_transform(graph)

    @property
    def raw_file_names(self):
        return ['cicy3o.h5']

    @property
    def processed_file_names(self):
        return ['cicy3o.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pyg.data.download_url(self.url, self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        path = os.path.join(self.raw_dir,'cicy3o.h5')
        dataframe = pd.read_hdf(path)
        dataframe['matrix'] = dataframe['matrix'].apply(lambda x: torch.tensor(x,dtype=torch.float))
        dataframe = dataframe.sample(frac=1)
        graphs_list = dataframe_to_list(dataframe)
        data_list = [graph_to_data(graph) for graph in graphs_list]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter_func(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform_func(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class FavorableCICY3(pyg.data.InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.url = 'http://www.lpthe.jussieu.fr/~erbin/files/data/cicy3f.h5'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def pre_filter_func(self,graph):
        if isinstance(self.pre_filter,list):
            for check in self.pre_filter:
                if not check(graph):
                    return False
            return True
        return self.pre_filter(graph)

    def pre_transform_func(self,graph):
        if isinstance(self.pre_transform,list):
            for transform in self.pre_transform:
                graph = transform(graph)
            return graph
        return self.pre_transform(graph)

    @property
    def raw_file_names(self):
        return ['cicy3f.h5']

    @property
    def processed_file_names(self):
        return ['cicy3f.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pyg.data.download_url(self.url, self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        path = os.path.join(self.raw_dir,'cicy3f.h5')
        dataframe = pd.read_hdf(path)
        dataframe['matrix'] = dataframe['matrix'].apply(lambda x: torch.tensor(x,dtype=torch.float))
        dataframe = dataframe.sample(frac=1)
        graphs_list = dataframe_to_list(dataframe)
        data_list = [graph_to_data(graph) for graph in graphs_list]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter_func(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform_func(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def convert_to_lightning(dataset,batch_size=32):
    return pyg.data.LightningDataset(dataset,batch_size=batch_size)
