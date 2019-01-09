import random
import os
import pickle

class Data:
	def __init__(self,config):
		self.config = config
		if not os.path.exists('./data/lucaweihs/processedData.pkl') or config.getboolean('general','reprocess'):
			self.splitData(config)
		else:
			self.loadData() 

	def splitData(self,config):
		def getData(path, seed, testRatio, header = False):
			whole = []
			with open(path) as f:
				for idx,line in enumerate(f):
					if header and idx==0 : continue
					whole.append(map(lambda x:int(x.strip()), line.split('\t')))
					if idx > 200 and config.getboolean('general','debug') : break
			random.seed(seed)
			random.shuffle(whole)
			val,test,train = whole[:int(testRatio*len(whole))], whole[int(testRatio*len(whole)):2*int(testRatio*len(whole))], whole[2*int(testRatio*len(whole)):]
			return train, val, test
	
		self.train, self.val, self.test = getData(self.config['path']['paperHistoryPath'],int(config['general']['seed']), float(config['general']['testRatio']))
		self.trainTarget, self.valTarget, self.testTarget = getData(self.config['path']['paperResponsePath'], int(config['general']['seed']), float(config['general']['testRatio']),header=True)

		# save processed data
		with open('./data/lucaweihs/processedData.pkl','wb') as f:
			pickle.dump((self.train,self.val,self.test,self.trainTarget,self.valTarget,self.testTarget),f)

	def loadData(self):
		with open('./data/lucaweihs/processedData.pkl','rb') as f:
			self.train, self.val, self.test, self.trainTarget, self.valTarget, self.testTarget = pickle.load(f)