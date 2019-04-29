import os
import torch

### USER-SET CONFIG ###
LEARNING_RATE = 0.001
VALID_EVERY = 10
NUM_EPOCH = 100
BATCH_SIZE = 100
DATA_SOURCE = 'DBLP' # KDD, LUCA, DBLP
MODEL_TYPE = 'Many2One' # 'Many2Many', 'Many2One' ,'Many2OneAttention'
TEST_RATIO = 0.15
SEED = 3
EMBEDDING_DIM = 50
DEVICE = torch.device('cuda')

### HARD CONFIG ###
DATA_DIR = {'DBLP':os.path.join(os.getcwd(), 'data','DBLP'), 
			'KDD':os.path.join(os.getcwd(), 'data','KDD'), 
			'LUCA':os.path.join(os.getcwd(), 'data','lucaweihs')}
DATA_TO_USE = {'DBLP':['history','paper','author','response','paper_embedding'],
				'KDD':['history','paper','author','response'],
				'LUCA':['history','paper','author','response']}
NUM_FEATURES = {'history':2,
				'paper':5,
				'author':5,
				# 'venue':5,
				'response':10,
				'paper_embedding':7}

ID_PATH ={'DBLP':'PaperID-dblp-v2',
			'KDD':None,
			'LUCA':None}
HISTORY_PATH = {'DBLP':'history-dblp-v2',
				'KDD':None,
				'LUCA':None}
RESPONSE_PATH = {'DBLP':'response-dblp-v2',
				'KDD':None,
				'LUCA':None}