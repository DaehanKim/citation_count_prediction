import os
import random

import setting

def shuffled(lst):
	ret = list(lst)
	random.shuffle(ret)
	return ret

def dynamic_pad(batch_dict):
	to_pad = batch_dict['history']
	max_len = max([len(i) for i in to_pad])
	for i in range(len(to_pad)):
		to_pad[i] = [.0 for _ in range(max_len-len(to_pad[i]))] + to_pad[i]

class FeatureExtractor(object):
	def __init__(self, data_type):
		self.data_type = data_type
		self.dir_path = setting.DATA_DIR[data_type]
		self.extract()
	
	@staticmethod
	def extract_file(file_path):
		with open(file_path) as f: ret = f.readlines()
		ret = [[float(j) for j in i.split('\t')] for i in ret]
		return ret

	def extract(self):
		file_path = os.path.join(self.dir_path, 'processed_features')
		self.data_dict = {}
		for dtype in setting.DATA_TO_USE[self.data_type]:
			self.data_dict[dtype] = FeatureExtractor.extract_file(os.path.join(file_path,dtype))
		
class Loader(object):
	def __init__(self):
		self.data_dict = FeatureExtractor(setting.DATA_SOURCE).data_dict
		self.split_data_dict = {k : self.get_split_data(k) for k in ['train','valid','test']}

	def get_split_data(self, which = 'train'):		
		ret_dict = {}
		for k,v in self.data_dict.items():
			random.seed(setting.SEED)
			shuffled_list = shuffled(v)
			division_index = int(setting.TEST_RATIO * len(shuffled_list))
			split = {'train': shuffled_list[division_index*2:],
					 'valid': shuffled_list[:division_index],
					 'test':  shuffled_list[division_index:2*division_index]}
			ret_dict[k] = split[which]
		return ret_dict

	def get_data_iter(self, which = 'train'):
		concerned_data = self.split_data_dict[which]
		num_batch = len(concerned_data['history'])//setting.BATCH_SIZE
		for i in range(1,num_batch+1):
			start_idx = (i-1)*setting.BATCH_SIZE
			end_idx = i*setting.BATCH_SIZE
			ret = {k:v[start_idx:end_idx] for k,v in concerned_data.items()}
			dynamic_pad(ret)
			yield ret