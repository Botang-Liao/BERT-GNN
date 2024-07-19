from model import CustomBERTModel, Tokenizer
from trainingManager import TrainingManager
from datasetManager import DatasetManager, PatentClassificationDatasetColumnChooser



if __name__ == "__main__":

    model = CustomBERTModel()
    tokenizer = Tokenizer()
    model = model.load_sota_model()
    data_chooser = PatentClassificationDatasetColumnChooser()
    dataset = DatasetManager(data_chooser.get_abstract_and_claim_data())
    tester = TrainingManager(model, tokenizer, dataset)
    tester.validation()
    tester.test()
    
    
    