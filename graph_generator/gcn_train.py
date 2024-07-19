from model import GAT, GCN
from trainingManager import Trainer
from graphManager import GraphManager, GraphConnection
import numpy as np


val_data = np.load('/HDD/n66104571/patent_classification/data/val_data.npy')
val_label = np.load('/HDD/n66104571/patent_classification/data/val_label.npy')
test_data = np.load('/HDD/n66104571/patent_classification/data/test_data.npy')
test_label = np.load('/HDD/n66104571/patent_classification/data/test_label.npy')

data = np.concatenate([np.eye(47), val_data, test_data], axis=0)
label = np.concatenate([np.eye(47), val_label, test_label], axis=0)
connection = GraphConnection().connect_by_bert(np.concatenate([val_label, test_label], axis=0))
dataset_manager = GraphManager(data, label, connection)
model = GCN().load_sota_model()
trainer = Trainer(model, dataset_manager.graph)
trainer.train()
trainer.obervation()