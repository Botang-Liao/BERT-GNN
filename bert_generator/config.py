import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = '/HDD/n66104571/patent_classification/preprocess_df.csv'

BERT_MODEL_PATH = '/HDD/n66104571/patent_classification/model/finetune_bert_model'

IMG_PATH = '/HDD/n66104571/patent_classification/img'

OUTPUT_PATH = '/output'

RANDOM_STATE = 42

LR = 1e-06
EPOCHES = 1
BATCH_SIZE = 8

TRAIN_DATA_RATE = 0.8
VALIDATION_DATA_RATE = 0.1
TEST_DATA_RATE = 0.1

THRESHOLD = 0.5

