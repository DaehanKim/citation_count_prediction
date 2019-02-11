import math
import model
import torch
import string
import random


### neural net utils ###

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def loadModel(checkpoint):
	with open(checkpoint,'rb') as f: state_dict, hidden, args = torch.load(f)
	model_ = model.basicLSTM(args)
	model_.load_state_dict(state_dict)
	return model_,hidden

### functions ###

def cnt2category(cnt):
	'''wraps citation counts into its category'''
	log2 = lambda x: math.log10(x)/math.log10(2)
	return log2(cnt) + 1 if cnt != 0 else 0

def write_result(mape, R2,args, path = 'result.txt'):
	'''mape, R2 in list'''
	with open(path,'w') as f:
		f.write('mape : %r\nR2 : %r\nArgs : %r\n'%(mape,R2,args))

### abstract process ###

def extractTitleAndAbsKDD(fpath):
    '''KDD dataset'''
    with open(fpath) as f: abst =  f.read()
    title = abst.split('\\\\')[1].split('Title:')[1].split('Authors')[0].strip()
    abstract = abst.split('\\\\')[2].strip()
    title = title.translate(None, string.punctuation)
    abstract = abstract.translate(None, string.punctuation)
    return title, abstract

def extractAbstractDBLP(fpath):
    with open(fpath) as f: abst =  f.read()
    return abst

def load_pretrained_doc2vec(fpath, seed, testRatio):
    with open(fpath) as f: vectors = [eval(line) for line in f.readlines()]
    random.seed(seed)
    random.shuffle(vectors)
    if testRatio > 1.0:
        testIdx = int(testRatio)
    else:
        testIdx = int(testRatio*len(vectors))
    val, test, train = vectors[:testIdx], vectors[testIdx:testIdx*2], vectors[testIdx*2:]
    return val, test, train
