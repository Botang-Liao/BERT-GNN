import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import DATA_DIR, RANDOM_STATE, TRAIN_DATA_RATE, VALIDATION_DATA_RATE, TEST_DATA_RATE, BATCH_SIZE


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
    
    def get_abstract_and_claim_data(self):
        data = self.df['abstract&claim']
        label = self.df.iloc[:, 4:-1]
        print(data.shape, label.shape)
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

