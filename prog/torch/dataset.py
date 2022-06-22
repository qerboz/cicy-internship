import torch
import torch_geometric as pyg

def mat_to_graph(matrices):
    if type(matrices) == torch.Tensor:
        nodes_features = []
        edges_list = []
        edges_features = []
        rows = matrices.shape[0]
        columns = matrices.shape[1]
        nodes_features += [[0,1]]*rows + [[1,0]]*columns
        for i in range(rows):
            for j in range(columns):
                if matrices[i][j] > 0:
                    edges_list.append([i,rows+j])
                    edges_features.append(matrices[i][j])
        edges_list = torch.transpose(torch.tensor(edges_list,dtype=torch.int),0,1)
        edges_features = torch.tensor(edges_features,dtype=torch.float)
        sparse_adj = torch.sparse_coo_tensor(edges_list,torch.tensor([1]*len(edges_features)),size=(len(nodes_features),len(nodes_features)))
        sparse_adj += torch.transpose(sparse_adj,0,1)
        mod_sparse_adj = torch.sparse_coo_tensor(edges_list,edges_features,size=(len(nodes_features),len(nodes_features)))
        return {'nodes':nodes_features,'edges_list':edges_list,'edges_features':edges_features,'adj_mat':sparse_adj,'mod_adj_mat':mod_sparse_adj}
    else:
        graph_list = []
        for mat in matrices:
            graph_list.append(mat_to_graph1(mat))
        return graph_list

def mat_to_graph2(matrices):
    if type(matrices) == np.ndarray:
        nodes_features = []
        edges_list = []
        edges_features = []
        rows = matrices.shape[0]
        columns = matrices.shape[1]
        node_num_mat = np.zeros(matrices.shape,dtype='float32')
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
        edges_list,edges_features = np.array(edges_list),np.array(edges_features)
        sparse_adj = sp.sparse.coo_matrix(([1]*len(edges_features),(edges_list[:,0],edges_list[:,1])),shape=(len(nodes_features),len(nodes_features)))
        return {'nodes':nodes_features,'edges_list':edges_list,'edges_features':edges_features,'adj_mat':sparse_adj}
    else:
        graph_list = []
        for mat in matrices:
            graph_list.append(mat_to_graph2(mat))
        return graph_list

def graph_to_mat1(graph):
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

def check_graph_mat1(graph,mat):
    return (graph_to_mat1(graph)==mat).all()

def df_to_list(df):
    graphs_list = mat_to_graph1(list(df['matrix'].values))
    for i in range(len(graphs_list)):
        graphs_list[i]['target'] = {'h11':df['h11'].values[i],'h21':df['h21'].values[i]}
    return graphs_list