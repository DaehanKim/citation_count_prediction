import torch
import new_models
import loader
import setting
import sys

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

### load data ###
data_loader = loader.Loader()

# mapeList = []
# rsquaredList = []
### model init / load ###
# for numExperiment in range(0,10):
model = getattr(new_models, setting.MODEL_TYPE)().to(setting.DEVICE)

### train ###
for epoch in range(1,setting.NUM_EPOCH+1):
	for idx, batch in enumerate(data_loader.get_data_iter('train')):
		model.fit(batch)
		if (idx+1) % setting.VALID_EVERY == 0: 
			mape, r2 = model.validate(data_loader.get_data_iter('valid'))
			sys.stdout.write('valid EPOCH{:3d} BATCH{:3d} | mape : {:.3f}, r2 : {:.3f}\r'.format(epoch,idx+1,mape[-1],r2[-1]))

### test ###
r2, mape = model.validate(data_loader.get_data_iter('test'),make_plot=True)
# rsquaredList.append(r2[-1])
# mapeList.append(mape[-1])
print ''
print 'test | mape : {}, r2 : {}'.format(r2[-1], mape[-1])

# print 'average test | mape: {}, r2: {}'.format(mean(mapeList), mean(rsquredList))