from sklearn.metrics import r2_score
import torch
import numpy as np

def calc_mape(pred, target):
	return np.absolute((pred-target)/target).mean(axis=0)

class Scorer(object):
	def __init__(self):
		self.pred_list = []
		self.target_list = []
	def update(self, pred, target):
		self.pred_list += pred.tolist()
		self.target_list += target.tolist()
	def reset(self):
		self.pred_list = []
		self.target_list = []
	def compute_score(self):
		self.mape = calc_mape(np.array(self.pred_list), np.array(self.target_list))
		self.r2 = r2_score(np.array(self.pred_list), np.array(self.target_list), multioutput='raw_values')
		return self.mape, self.r2