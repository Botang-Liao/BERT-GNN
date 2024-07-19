import torch
import torch_geometric
import torch.nn as nn
from config import DEVICE, LR, WEIGHT_DECAY, EPOCH, THRESHOLD, IMG_PATH
from lossFunction import multilabel_categorical_crossentropy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model:torch.nn.Module, graph:torch_geometric.data.Data):
        self.model = model.to(DEVICE)
        self.graph = graph.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=EPOCH)
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = multilabel_categorical_crossentropy
        self.test_mask_index = torch.where(self.graph.test_mask)[0]

    def train(self):
        train_loss = []
        val_loss = []
        val_acc = []
        val_f1 = []
        val_recall = []
        val_precision = []
        rec_freq = 500
        best_result = 0
        pos_weight = 1
        self.model.train()
        for epoch in range(EPOCH+1):
            self.optimizer.zero_grad()
            out = self.model(self.graph)
            #loss = self.criterion(out[self.graph.train_mask], self.graph.y[self.graph.train_mask])
            loss = self.criterion(out[self.graph.train_mask], self.graph.y[self.graph.train_mask], pos_weight=pos_weight)
            self.graph.train_mask = self.update_training_mask(self.graph.train_mask.shape[0], 0.01)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if epoch % rec_freq == 0:
                
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
                info = self.obervation()
                val_acc.append(info["accuracy"])
                val_f1.append(info["f1 score"])
                val_recall.append(info["recall"])
                val_precision.append(info["precision"])
                train_loss.append(loss.item())
                val_loss.append(info["loss"])
                pos_weight = info['precision'] / info['recall']
                if info["f1 score"] >= best_result:
                    print('new accuracy :', info["f1 score"], 'best accuracy :', best_result)
                    print('update parameter') 
                    self.model.save_model()
                    best_result = info["f1 score"]
                else:
                    print('new accuracy :', info["f1 score"], 'best accuracy :', best_result)
                    print('keep original parameter')
                print('pos_weight:', pos_weight)
            
        self.plot_fig(train_loss, 'train_loss', 'epoch', 'loss', IMG_PATH + '/gcn_train_loss.png')
        self.plot_fig(val_loss, 'validation_loss', 'epoch', 'loss', IMG_PATH + '/gcn_validation_loss.png') 
        self.plot_fig(val_acc, 'val_acc', 'epoch', 'acc', IMG_PATH + '/gcn_val_acc.png')
        self.plot_fig(val_f1, 'val_f1', 'epoch', 'f1', IMG_PATH + '/gcn_val_f1.png')
        self.plot_fig(val_recall, 'val_recall', 'epoch', 'recall',  IMG_PATH + '/gcn_val_recall.png')
        self.plot_fig(val_precision, 'val_precision', 'epoch', 'precision', IMG_PATH + '/gcn_val_precision.png')
        self.plot_fig(train_loss, 'loss', 'epoch', 'loss', IMG_PATH + '/gcn_total_loss.png', other_file=val_loss)
        print('success to save figs')
    
    def obervation(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.graph)
            pred = pred[self.graph.test_mask]
            loss = self.criterion(pred, self.graph.y[self.graph.test_mask])
            pred = (torch.sigmoid(pred) >= THRESHOLD)
            y_pred = pred.cpu().numpy()
            y_true = self.graph.y[self.graph.test_mask].cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, zero_division=1, average='macro')
            precision = precision_score(y_true, y_pred, zero_division=1,average='macro')
            f1 = f1_score(y_true, y_pred, zero_division=1, average='macro')
            print('accuracy: %f / f1: %f / recall: %f / precision: %f' % (accuracy, f1, recall, precision))
            return {"accuracy" : accuracy,
                    "f1 score" : f1,
                    "recall" : recall,
                    "precision" : precision,
                    "loss" : loss.item()
                }   
             
    def predict(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.graph)
            pred = logits.max(dim=1)[1]
            correct = float(pred[self.graph.test_mask].eq(self.graph.y[self.graph.test_mask]).sum().item())
            acc = correct / self.graph.test_mask.sum().item()
            print(f'Test Accuracy: {acc}')
            
    def update_training_mask(self, size, percentage=0.01):
        num_trues = int(size * percentage)
        # 生成全为 False 的 tensor
        tensor = torch.zeros(size, dtype=torch.bool)
        # 随机选择位置设置为 True
        indices = torch.randperm(size)[:num_trues]
        indices[0] = 4
        indices = torch.tensor(list(set(indices.numpy()) - set(self.test_mask_index.cpu().numpy())))
        tensor[indices] = True
        return tensor
    
    def plot_fig(self, file, title, x_label, y_label, save_path, other_file=None):
        plt.plot(file)
        if other_file:
            plt.plot(other_file)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(save_path)
        plt.close()