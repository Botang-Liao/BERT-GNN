from config import DEVICE, BERT_MODEL_PATH, LR, EPOCHES, THRESHOLD, IMG_PATH
from datasetManager import DatasetManager
from torch.utils.data import DataLoader
from model import Tokenizer
import torch
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from lossFunction import multilabel_categorical_crossentropy
from transformers import get_linear_schedule_with_warmup

class TrainingManager:
    def __init__(self, model : nn.Module, tokenizer : Tokenizer, datasetManager : DatasetManager):
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.datasetManager = datasetManager
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        #self.criterion = multilabel_categorical_crossentropy
        self.criterion = nn.BCEWithLogitsLoss()
        total_number = 193603*EPOCHES
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_number)
        
    def train(self):
        # Enter Train Mode
        train_loss = []
        validation_loss = []
        val_acc = []
        val_f1 = []
        val_recall = []
        val_precision = []
        best_result = 0.826687
        update_times = 0
        
        for epoch in range(EPOCHES):
            self.model.train()
            this_epoch_value = 0  
            print('epoch:', epoch+1 ,'/', EPOCHES)
            for idx, (sentence, label) in enumerate(self.datasetManager.train_dataloader):
            #for (sentence, label) in tqdm(self.datasetManager.train_dataloader):
            # for (sentence, label) in self.datasetManager.train_dataloader:
                input_ids, attention_mask = self.tokenizer.tokenize(sentence)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                label = label.to(DEVICE)
                pred = self.model(input_ids, attention_mask)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                this_epoch_value += loss.item()
                if idx % 1000 == 0:
                    print(idx, '/', len(self.datasetManager.train_dataloader))
                
            train_loss.append(this_epoch_value)    
            info = self.validation()
            validation_loss.append(info["loss"])
            val_acc.append(info["accuracy"])
            val_f1.append(info["f1 score"])
            val_recall.append(info["recall"])
            val_precision.append(info["precision"])
            if info["f1 score"] >= best_result:
                print('new accuracy :', info["f1 score"], 'best accuracy :', best_result)
                print('update parameter')
                best_result = info["f1 score"]
                update_times += 1
                self.model.save_model()
            else:
                print('new accuracy :', info["f1 score"], 'best accuracy :', best_result)
                print('keep original parameter')

            
            
        self.plot_fig(train_loss, 'train_loss', 'epoch', 'loss', IMG_PATH + '/bert_train_loss.png')
        self.plot_fig(validation_loss, 'validation_loss', 'epoch', 'loss', '/bert_validation_loss.png') 
        self.plot_fig(train_loss, 'total_loss', 'epoch', 'loss', IMG_PATH + '/bert_total_loss.png', validation_loss)
        self.plot_fig(val_acc, 'val_acc', 'epoch', 'acc',  IMG_PATH + '/bert_val_acc.png')
        self.plot_fig(val_f1, 'val_f1', 'epoch', 'f1',  IMG_PATH + '/bert_val_f1.png')
        self.plot_fig(val_recall, 'val_recall', 'epoch',  IMG_PATH + 'recall', '/bert_val_recall.png')
        self.plot_fig(val_precision, 'val_precision',  IMG_PATH + 'epoch', 'precision', '/bert_val_precision.png')
        print('success to save figs')
        print('update times :', update_times)
        
    def validation(self):
        return self.inference(self.datasetManager.validation_dataloader)
       
    def test(self):
        return self.inference(self.datasetManager.test_dataloader) 
    
    def inference(self, dataloader : DataLoader):
        labels = []
        preds = []
        total_loss = 0
        self.model.eval()  # 将模型设置为评估模式
        with torch.no_grad(): 
            #for (sentence, label) in tqdm(dataloader):
            for (sentence, label) in dataloader:
                input_ids, attention_mask = self.tokenizer.tokenize(sentence)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                label = label.to(DEVICE)
                pred = self.model(input_ids, attention_mask)
                total_loss += self.criterion(pred, label).item()
                pred = (torch.sigmoid(pred) >= THRESHOLD)
                preds.extend(pred.cpu().numpy())
                labels.extend(label.cpu().numpy())

        
        y_pred = np.concatenate(preds, axis=0).reshape(-1, 47)
        y_true = np.concatenate(labels, axis=0).reshape(-1, 47)
        
        # print(y_pred.shape)
        # print(y_true.shape)
        # print(y_pred[0])
        # print(y_true[0])
        # print('-'*20)
        # print(y_pred[1])
        # print(y_true[1])
        # print('-'*20)
        # print(y_pred[2])
        # print(y_true[2])
        # print('-'*20)
        # print(y_pred[3])
        # print(y_true[3])
        
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=1, average='macro')
        precision = precision_score(y_true, y_pred, zero_division=1,average='macro')
        f1 = f1_score(y_true, y_pred, zero_division=1, average='macro')
        print('accuracy: %f / f1: %f / recall: %f / precision: %f' % (accuracy, f1, recall, precision))
        return {"accuracy" : accuracy,
                "f1 score" : f1,
                "recall" : recall,
                "precision" : precision,
                "loss" : total_loss
                }   

        
    def plot_fig(self, file, title, x_label, y_label, save_path, other_file=None):
        plt.plot(file)
        if other_file:
            plt.plot(other_file)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(save_path)
        plt.close()
        
    def predict(self, dataloader : DataLoader):
        labels = []
        preds = []
        total_loss = 0
        self.model.eval()  # 将模型设置为评估模式
        with torch.no_grad(): 
            for (sentence, label) in tqdm(dataloader):
                input_ids, attention_mask = self.tokenizer.tokenize(sentence)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                label = label.to(DEVICE)
                pred = torch.sigmoid(self.model(input_ids, attention_mask))
                preds.extend(pred.cpu().numpy())
                labels.extend(label.cpu().numpy())

        return np.concatenate(preds, axis=0).reshape(-1, 47), np.concatenate(labels, axis=0).reshape(-1, 47)
    
            

    def predict_train(self):
        return self.predict(self.datasetManager.train_dataloader)
    
    def predict_validation(self):
        return self.predict(self.datasetManager.validation_dataloader)    
    
    def predict_test(self):
        return self.predict(self.datasetManager.test_dataloader)