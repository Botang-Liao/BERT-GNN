from model import CustomBERTModel, Tokenizer
from trainingManager import TrainingManager
from datasetManager import DatasetManager, PatentClassificationDatasetColumnChooser



if __name__ == "__main__":
    model = CustomBERTModel()
    model = model.load_sota_model()
    tokenizer = Tokenizer()
    data_chooser = PatentClassificationDatasetColumnChooser()
    dataset = DatasetManager(data_chooser.get_abstract_claim())
    trainer = TrainingManager(model, tokenizer, dataset)
    trainer.train()
    trainer.test()
    # model = CustomBERTModel()
    # model = model.load_model()
    # tokenizer = Tokenizer()
    # data_chooser = PatentClassificationDatasetColumnChooser()
    # dataset = DatasetManager(data_chooser.get_abstract_and_claim_data())
    # trainer = TrainingManager(model, tokenizer, dataset)
    # trainer.test()
    
    
    