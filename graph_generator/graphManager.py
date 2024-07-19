import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import numpy as np
import torch
from datasetManager import DatasetManager, PatentClassificationDatasetColumnChooser
from tqdm import tqdm
import numpy as np


class GraphConnection:
    def __init__(self):
        self.num = 47
        self.list_first_elements = []
        self.list_second_elements = []

    def connect(self, labels): 
        indices = np.where(labels == 1)
        self.list_first_elements = (indices[0] + self.num).tolist()
        self.list_second_elements = indices[1].tolist()
        return torch.tensor([self.list_first_elements+self.list_second_elements, self.list_second_elements+self.list_first_elements])
                  
    def connect_by_bert(self, data, num=5):

        # 對每一行應用 np.argsort() 函數，並且使用 [::-1] 來反轉索引，讓它成為從大到小的排序
        # 然後取每行的前五個最大值的索引
        top_n_indices = np.argsort(data, axis=1)[:, ::-1][:, :num]
        for idx, vals in enumerate(top_n_indices):
            for val in vals:
                self.list_first_elements.append(idx + self.num)
                self.list_second_elements.append(val)
        return torch.tensor([self.list_first_elements+self.list_second_elements, self.list_second_elements+self.list_first_elements])
                  

class GraphManager:
    def __init__(self, data, label, connection, percentage=0.1):
        self.train_mask ,self.test_mask = self.create_non_overlapping_tensors(data.shape[0])
        self.graph = self.create_graph(data, label, connection)
       
    def create_graph(self, data, label, connection):
        node_features = torch.from_numpy(data).float()
        edge_index = connection  
        labels = torch.from_numpy(label)
        #print('node_features:', node_features.shape) 
        #print('edge_index:', edge_index.shape)
        #print('labels:', labels.shape)
        #print(type(node_features), type(edge_index), type(labels))
        
        G = Data(x=node_features, edge_index=edge_index, y=labels, train_mask=self.train_mask, test_mask=self.test_mask)
        return G
    
    def plot_graph(self):
      
        # 转换为 networkx 图
        nx_graph = to_networkx(self.graph, to_undirected=True)

        # 准备绘图
        pos = nx.spring_layout(nx_graph)  # 使用 Spring 布局
        plt.figure(figsize=(12, 6))

        # 绘制训练图
        plt.subplot(121)
        train_nodes = [i for i in range(len(self.train_mask)) if self.train_mask[i]]
        nx.draw(nx_graph, pos, node_color=['skyblue' if self.graph.y[i] == 0 else 'salmon' for i in train_nodes], 
                with_labels=True, nodelist=train_nodes, edge_color='gray')
        plt.title('Training Nodes')

        # 绘制测试图
        plt.subplot(122)
        test_nodes = [i for i in range(len(self.test_mask)) if self.test_mask[i]]
        nx.draw(nx_graph, pos, node_color=['skyblue' if self.graph.y[i] == 0 else 'salmon' for i in test_nodes], 
                with_labels=True, nodelist=test_nodes, edge_color='gray')
        plt.title('Testing Nodes')

        plt.savefig('graph.png')
        
    def create_non_overlapping_tensors(self, size, percentage=0.1):
        # 计算每个 tensor 中 1 的数量
        num_ones = int(size * percentage)

        # 用random seed 生成隨機索引
        all_indices = np.random.RandomState(seed=42).permutation(size)
        test = all_indices[:num_ones]
        val = all_indices[num_ones:]

        # 創建全為 0 的 tensor
        tensor = torch.zeros((2,size), dtype=torch.bool)

        # 在指定位置設置為 1
        tensor[0,val] = True
        tensor[1,test] = True
        

        return tensor
    
    
