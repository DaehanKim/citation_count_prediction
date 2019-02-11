from multiprocessing import Pool
import os

def runOneExperiment(args):
	modelType, encoderType, embedCategory, attention, bidir, repeat = args
	resultPath = nameResultPath(args)
	commend = "CUDA_VISIBLE_DEVICES=1 python lstmEncoderMain.py --learningRate 0.01 --numEpoch 7 --keepLessThan5 --validEvery 2000 --encoderType %s --modelType %s --resultPath %s"%(encoderType,modelType, resultPath)
	if embedCategory: commend += ' --embedCategory'
	if attention : commend += ' --attention'
	if bidir : commend += ' --bidir'
	if encoderType == 'None' : commend += " --abstractDir ''"

	os.system(commend)

def nameResultPath(args):
	modelType, encoderType, embedCategory, attention,bidir, repeat = args
	resultPath = 'results/dblp/%s_%sencoder'%(modelType,encoderType)
	if bidir: resultPath += 'Bidir'
	if embedCategory: resultPath += '_catemb'
	if attention : resultPath += '_att'
	resultPath += '_%d.txt'%(repeat)
	return resultPath

# experiment settings
argList = []

# for modelType, encoderType in [('GRU','LSTM'),('LSTM','GRU')]:
for repeat in range(5):
	for modelType in ('LSTM','GRU'):
		for encoderType in ('GRU','LSTM','None'):
			for embedCategory, attention in [(True,True),(True,False),(False,False)]:
				for bidir in (True, False):
					arg = (modelType, encoderType, embedCategory, attention,bidir,repeat)
					if nameResultPath(arg).split('/')[-1] in os.listdir('results/kdd/'):
						continue
					if bidir and encoderType == 'None': continue
					argList += [arg]

# run all experiments
p = Pool(4)
_ = p.map(runOneExperiment,argList)

# runOneExperiment(argList[0])