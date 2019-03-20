import torch
import new_models
import loader
import setting

### load data ###
data_loader = loader.Loader()

### model init / load ###
model = getattr(new_models, setting.MODEL_TYPE)().to(setting.DEVICE)

### train ###
for epoch in range(1,setting.NUM_EPOCH+1):
	for idx, batch in enumerate(data_loader.get_data_iter('train')):
		model.fit(batch)
		if (idx+1) % setting.VALID_EVERY == 0: 
			print 'valid - EPOCH={} BATCH={} | mape : {}, r2 : {}'.format(epoch,idx+1,*model.validate(data_loader.get_data_iter('valid')))

### test ###
r2, mape = model.validate(data_loader.get_data_iter('test'))
print 'test | mape : {}, r2 : {}'.format(r2, mape)