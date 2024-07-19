import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch.nn import Linear

# fundation of GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(47, 1024)
        self.conv2 = GCNConv(1024, 1024)
        self.conv3 = Linear(1024, 47)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x)

        return x
    
    def save_model(self):
        torch.save(self,'/HDD/n66104571/patent_classification/model/gcn_param.pt')
        
    def load_model(self):
        model = torch.load('/HDD/n66104571/patent_classification/model/gcn_param.pt')
        return model
    
    def load_sota_model(self):
        model = torch.load('/HDD/n66104571/patent_classification/model/sota_in_experiment/gcn_param.pt')
        return model
    
class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(47, 128, heads=8, dropout=0.5)
        self.conv2 = GATConv(1024, 128, heads=8, dropout=0.5)
        self.conv3 = Linear(1024, 47)
       

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x)
        return x
    
    def save_model(self):
        torch.save(self,'/HDD/n66104571/patent_classification/model/gat_param.pt')
        
    def load_model(self):
        model = torch.load('/HDD/n66104571/patent_classification/model/gat_param.pt')
        return model
    
    def load_sota_model(self):
        model = torch.load('/HDD/n66104571/patent_classification/model/sota_in_experiment/gat_param.pt')
        return model