import random
import os
import pickle

from torchtext import data

import util

class Data:
	def __init__(self,config):
		self.config = config
		self.field = data.Field(lower = True, tokenize = 'spacy')
		# if not os.path.exists('./data/lucaweihs/processedData.pkl') or config.reprocess:
		# 	self.splitData(config)
		# else:
		# 	self.loadData()
		self.splitData(config)

	def splitData(self,config):
		def getData(path, seed, testRatio, header = False):
			whole = []
			with open(path) as f:
				for idx,line in enumerate(f):
					if header and idx==0 : continue
					whole.append(map(lambda x:int(x.strip()), line.split('\t')))
					if idx > (6001 if header else 6000) and config.debug : break
			random.seed(seed)
			random.shuffle(whole)
			if testRatio >= 1.0: 
				testIdx = int(testRatio)
			else: 
				testIdx = int(testRatio*len(whole))
			val,test,train = whole[:testIdx], whole[testIdx:2*testIdx], whole[2*testIdx:]
			return train, val, test

		def getAbstract(dirPath, seed, testRatio):
			examples = []
			with open(os.path.join(os.getcwd(), 'data/DBLP/id_compatible_with_abstract.txt')) as f:
				ids = [line.strip() for line in f.readlines()]
			for idx,id_ in enumerate(ids):
				fpath = os.path.join(dirPath,id_+'.abs')
				abstract = util.extractAbstractDBLP(fpath)
				examples.append(data.Example.fromlist([abstract],[('text',self.field)]))
				if idx > 6000 and config.debug: break
			random.seed(seed)
			random.shuffle(examples)
			if testRatio >= 1.0: 
				testIdx = int(testRatio)
			else: 
				testIdx = int(testRatio*len(examples))
			val,test,train = examples[:testIdx], examples[testIdx:2*testIdx], examples[2*testIdx:]
			trainset,valset, testset = data.Dataset(train, {'text':self.field}), data.Dataset(val, {'text':self.field}), data.Dataset(test, {'text':self.field})
			self.field.build_vocab(trainset, min_freq=2)
			return data.Iterator.splits((trainset,valset,testset),(1,1,1),shuffle=False)

		self.train, self.val, self.test = getData(self.config.paperHistoryPath,self.config.seed, self.config.testRatio)
		self.trainTarget, self.valTarget, self.testTarget = getData(self.config.paperResponsePath,self.config.seed, self.config.testRatio, header = True)
		self.trainSource, self.valSource, self.testSource = (self.train, self.trainTarget) ,(self.val, self.valTarget), (self.test, self.testTarget)
		if len(config.abstractDir) != 0 :
			self.trainAbs, self.valAbs, self.testAbs = getAbstract(self.config.abstractDir, self.config.seed, self.config.testRatio)
			self.trainSource += (self.trainAbs,)
			self.valSource += (self.valAbs,)
			self.testSource += (self.testAbs,)


		# save processed data
		# with open('./data/lucaweihs/processedData.pkl','wb') as f:
		# 	pickle.dump(self,f)

	# def loadData(self):
	# 	with open('./data/lucaweihs/processedData.pkl','rb') as f:
	# 		self = pickle.load(f)

