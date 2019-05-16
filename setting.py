import os
import torch

### USER-SET CONFIG ###
LEARNING_RATE = 0.001
VALID_EVERY = 20
NUM_EPOCH = 20
BATCH_SIZE = 100
DATA_SOURCE = 'DBLP' # KDD, LUCA, DBLP
MODEL_TYPE = 'Many2One' # 'Many2Many', 'Many2One' ,'Many2OneAttention'
# MODEL_TYPE = 'Many2Many'
TEST_RATIO = 0.15
SEED = 1
EMBEDDING_DIM = 50
DEVICE = torch.device('cuda')

### HARD CONFIG ###
DATA_DIR = {'DBLP':os.path.join(os.getcwd(), 'data','DBLP'), 
			'KDD':os.path.join(os.getcwd(), 'data','KDD'), 
			'LUCA':os.path.join(os.getcwd(), 'data','lucaweihs')}
DATA_TO_USE = {'DBLP':['history-delta','paper','author','response-delta','venue-2feature'],
				'KDD':['history-delta','paper','author','venue','response-delta'],
				'LUCA':['history-delta','paper','author','response-delta','venue']}
NUM_FEATURES = {'history':2,
				'paper':5,
				'author':7,
				'venue':5,
				'response':10,
				'paper_embedding':7}

# ID_PATH ={'DBLP':'PaperID-dblp-v2',
# 			'KDD':None,
# 			'LUCA':None}
ID_PATH ={'DBLP':'id_larger_than_5_compatible_with_abstract_v2',
			'KDD':'PaperID-kdd-v2',
			'LUCA':None}
HISTORY_PATH = {'DBLP':'history_larger_than_5_compatible_with_abstract_v2',
				'KDD':'history-kdd-v2',
				'LUCA':None}
RESPONSE_PATH = {'DBLP':'response_larger_than_5_compatible_with_abstract_v2',
				'KDD':'response-kdd-v2',
				'LUCA':None}