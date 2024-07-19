import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import DATA_DIR, RANDOM_STATE, TRAIN_DATA_RATE, VALIDATION_DATA_RATE, TEST_DATA_RATE, BATCH_SIZE
import nltk
import re

class PatentClassificationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, label: pd.DataFrame):
        # if not isinstance(data, pd.DataFrame):
        #     raise ValueError("Data must be a pandas DataFrame.")
        # if not isinstance(label, pd.DataFrame):
        #     raise ValueError("Label must be a pandas DataFrame.")
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        try:
            sentence = self.data.loc[idx]
            labels = self.label.loc[idx]
            labels = np.array(labels, dtype=float)
            return sentence, labels
        except KeyError as e:
            raise KeyError(f"The required column is missing from the data: {str(e)}")


class PatentClassificationDatasetColumnChooser:
    def __init__(self):
        self.df = pd.read_csv(DATA_DIR)
    
    def extract_noun(self, claims):
        words = nltk.word_tokenize(claims)
        tags = nltk.pos_tag(words) # 对单个字符进行标注
        NN = [s1 for (s1,s2) in tags if s2 in ['NN', 'NNP']]
        #对list列表的tags的两个变量进行判断（s1代表第一个变量，s2代表第二个变量）
        #提取出tags的NN和NNP单词。NN表示普通名词，NNP表示专有名词
        result = ' '.join(NN)
        return result
    
    def data_clean(self, claims):
        claims = str(claims)
        claims = claims.lower()
        claims = claims.replace('\n', ' ')
        claim = "".join(e for e in claims)
        #extract first claim
        claim = re.findall(r"\b\d+\s?(?:\.|\)|\:|-)\s?.+?\s*(?:\.\s*|$)", claim, flags=re.DOTALL)

        if claim:
            return claim[0]
        else:
            return claims
    
    def get_abstract(self):
        data = self.df['abstract']
        label = self.df.iloc[:, 4:-1]
        dataset = PatentClassificationDataset(data, label)
        return dataset
    
    def get_claim(self):
        data = self.df['claims'].apply(self.data_clean)
        label = self.df.iloc[:, 4:-1]
        dataset = PatentClassificationDataset(data, label)
        return dataset
    
    def get_title(self):
        data = self.df['title']
        label = self.df.iloc[:, 4:-1]
        dataset = PatentClassificationDataset(data, label)
        return dataset

    def get_abstract_claim(self):
        data = self.df['abstract&claim']
        label = self.df.iloc[:, 4:-1]
        dataset = PatentClassificationDataset(data, label)
        return dataset    

    def get_abstract_title(self):
        data = self.df['abstract'] + self.df['title']
        label = self.df.iloc[:, 4:-1]
        dataset = PatentClassificationDataset(data, label)
        return dataset
        
    def get_claim_title(self):
        
        data = self.df['claims'].apply(self.data_clean) + self.df['title']
        label = self.df.iloc[:, 4:-1]
        dataset = PatentClassificationDataset(data, label)
        return dataset
        
    def get_abstract_claim_title(self):
        
        data = self.df['abstract']+self.df['claims'].apply(self.data_clean)+self.df['title']
        label = self.df.iloc[:, 4:-1]
        dataset = PatentClassificationDataset(data, label)
        return dataset
    

class DatasetManager:
    def __init__(self, data: PatentClassificationDataset):
        train_data, temp_data = train_test_split(data, train_size= TRAIN_DATA_RATE, random_state=RANDOM_STATE)
        validation_data, test_data = train_test_split(temp_data, test_size=TEST_DATA_RATE/(VALIDATION_DATA_RATE+TEST_DATA_RATE), random_state=RANDOM_STATE)
        self.train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
        self.validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
        self.test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)

    def get_train_data_number(self):
        return len(self.train_dataloader.dataset)
    
    def get_validation_data_number(self):
        return len(self.validation_dataloader.dataset)  
    
    def get_test_data_number(self):
        return len(self.test_dataloader.dataset)
    

if __name__ == "__main__":
    print("test 1 :")
    num = 5
    data = pd.DataFrame({"sentence": [f"sentence{i}" for i in range(num * 2)]})
    label = pd.DataFrame({"label": [1, 0] * num})
    dataset = PatentClassificationDataset(data, label)
    dataset_manager = DatasetManager(dataset)
    print("Train data size:", dataset_manager.get_train_data_number())
    print("Validation data size:", dataset_manager.get_validation_data_number())
    print("Test data size:", dataset_manager.get_test_data_number())
    print("-" * 20)
    print("test 2 :")
    data_chooser = PatentClassificationDatasetColumnChooser()
    dataset = data_chooser.get_abstract_and_claim_data()
    dataset_manager = DatasetManager(dataset)
    print("Train data size:", dataset_manager.get_train_data_number())
    print("Validation data size:", dataset_manager.get_validation_data_number())
    print("Test data size:", dataset_manager.get_test_data_number())

