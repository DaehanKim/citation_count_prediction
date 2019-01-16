import math
import model
import torch


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

def write_result(mape, R2, path = 'result.txt'):
	'''mape, R2 in list'''
	with open(path,'w') as f:
		f.write('mape : %r\nR2 : %r\n'%(mape,R2))