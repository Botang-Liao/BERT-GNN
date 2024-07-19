import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = '/HDD/n66104571/patent_classification/preprocess_df.csv'

BERT_MODEL_PATH = '/HDD/n66104571/patent_classification/model/finetune_bert_model'

OUTPUT_PATH = '/output'

IMG_PATH = '/HDD/n66104571/patent_classification/img'

RANDOM_STATE = 42


BATCH_SIZE = 8

TRAIN_DATA_RATE = 0.8
VALIDATION_DATA_RATE = 0.1
TEST_DATA_RATE = 0.1

THRESHOLD = 0.5

LR = 1e-5

WEIGHT_DECAY = 5e-4

EPOCH = 10000

THRESHOLD = 0.5

PERCENTAGE = 0.1