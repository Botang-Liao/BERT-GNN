import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer
from config import DEVICE, THRESHOLD

class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def tokenize(self, sentences):
        encoding = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        return input_ids, attention_mask
    

class CustomBERTModel(nn.Module):
    def __init__(self, device='cpu'):
        super(CustomBERTModel, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.linear1 = nn.Linear(768, 192)  # 假设您有47个输出类别
        self.linear2 = nn.Linear(192, 47)  # 假设您有47个输出类别
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的输出
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        #x = self.sigmoid(x)
        return x

    def predict(self, sentences):
        self.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 禁用梯度计算
            logits = self.forward(sentences)
            predictions = logits
            #logits = torch.sigmoid(logits)
            #print(logits[-1])
            #predictions = (logits >= THRESHOLD)
        return predictions 
    
    def save_model(self):
        torch.save(self,'/HDD/n66104571/patent_classification/model/bert_param.pt')
        
    def load_model(self):
        model = torch.load('/HDD/n66104571/patent_classification/model/bert_param.pt')
        return model
    
    def load_sota_model(self):
        model = torch.load('/HDD/n66104571/patent_classification/model/sota_in_experiment/bert_param.pt')
        return model