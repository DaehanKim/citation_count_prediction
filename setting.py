import os
import torch

### USER-SET CONFIG ###
LEARNING_RATE = 0.001
VALID_EVERY = 100
NUM_EPOCH = 20
BATCH_SIZE = 30
DATA_SOURCE = 'DBLP' # KDD, LUCA, DBLP
MODEL_TYPE = 'Many2Many' # 'Many2Many', 'Many2One'
TEST_RATIO = 0.15
SEED = 3
DEVICE = torch.device('cuda')

### HARD CONFIG ###
DATA_DIR = {'DBLP':os.path.join(os.getcwd(), 'data','DBLP'), 
			'KDD':os.path.join(os.getcwd(), 'data','KDD'), 
			'LUCA':os.path.join(os.getcwd(), 'data','lucaweihs')}
DATA_TO_USE = {'DBLP':'history_paper_response',
				'KDD':'history_paper_author_venue_response',
				'LUCA':'history_paper_author_venue_response'}
NUM_FEATURES = {'history':2,'paper':5,'author':5,'venue':5,'response':10}