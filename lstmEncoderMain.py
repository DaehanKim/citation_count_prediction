import os
import math
import torch
import torch.nn as nn
from torchtext import data
from argparse import ArgumentParser

import data
import model
import util

EPOCH = 0

argparser = ArgumentParser()
argparser.add_argument('--numEpoch',type=int, default=10)
argparser.add_argument('--numLayer',type=int, default=1)
argparser.add_argument('--logEvery',type=int, default =1000)
argparser.add_argument('--learningRate',type=float, default = 0.01)
argparser.add_argument('--gradientClip',type=float, default = 0.25)
argparser.add_argument('--seed',type=int, default = 20190102)
argparser.add_argument('--validEvery',type=int, default = 10000, help='perform validation and anneal learning rate every this iterations')
argparser.add_argument('--testRatio',type=float, default = 0.15, help='if int, it means the number of test samples')
argparser.add_argument('--paperHistoryPath', type=str, default=os.path.join(os.getcwd(),'data/DBLP/history_compatible_with_abstract.txt'))
argparser.add_argument('--loadModelPath', type=str, default='', help = 'path to checkpoint to resume training')
argparser.add_argument('--resultPath', type=str, default='result.txt', help='name of result file')
argparser.add_argument('--save', type=str, default='model.pt', help = 'path to checkpoint to save model')
argparser.add_argument('--modelType', type=str, default='GRU', help = 'type of main module')
argparser.add_argument('--encoderType', type=str, default='GRU', help = 'type of text encoder')
argparser.add_argument('--evalMetric', type=str, default='mape', help = 'evaluation metric for which model is fitted')
argparser.add_argument('--paperResponsePath', type=str, default=os.path.join(os.getcwd(),'data/DBLP/response_compatible_with_abstract.txt'))
argparser.add_argument('--abstractDir', type=str, default=os.path.join(os.getcwd(),'data/DBLP/abstracts'))
argparser.add_argument('--keepLessThan5',action='store_true')
argparser.add_argument('--bidir',action='store_true')
argparser.add_argument('--embedCategory',action='store_true', help='Whether to use category embedding')
argparser.add_argument('--attention',action='store_true', help='Whether to use attention for category embedding')
argparser.add_argument('--debug',action='store_true', help='To test a functionality, use this flag for shorter running time')
# argparser.add_argument('--debug')

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(trainSource,valSource,args):
    #data load
    global model_
    if len(args.abstractDir) != 0:
        global encoder
    global hidden 
    global hiddenForAbstractEncoder
    global lr

    targetMean=torch.FloatTensor(trainSource[1]).mean(dim=0).to(device)
    mape = torch.zeros(1,targetMean.size(0)).to(device)
    R2Numer = torch.zeros(1,targetMean.size(0)).to(device)
    R2Denom = torch.zeros(1,targetMean.size(0)).to(device)
    loss = []
    cnt = 0
    try:
        for batchNum, features in enumerate(zip(*trainSource)):
            if (batchNum+1) % args.logEvery == 0:
                (sum(loss)/cnt).backward()
                torch.nn.utils.clip_grad_norm_(model_.parameters(), args.gradientClip)
                if len(args.abstractDir) != 0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.gradientClip)
                    torch.nn.utils.clip_grad_norm_(hiddenForAbstractEncoder, args.gradientClip)
                torch.nn.utils.clip_grad_norm_(hidden, args.gradientClip)
                for p in model_.parameters(): # learning model parameters
                    p.data.add_(-lr, p.grad)
                if len(args.abstractDir) != 0:
                    for p in encoder.parameters():
                        p.data.add_(-lr, p.grad)
                    encoder.zero_grad()
                    hiddenForAbstractEncoder = util.repackage_hidden(hiddenForAbstractEncoder)
                model_.zero_grad()
                hidden = util.repackage_hidden(hidden)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                R2 = 1 - R2Numer/R2Denom
                print 'epoch %d | batch %d/%d | lr : %.6f | cnt : %d | train MAPE : %r | train R2 : %r'%(EPOCH,batchNum+1,len(trainSource[1]), lr ,cnt,(mape/cnt).flatten().tolist(), R2.flatten().tolist())
                print out[:,:-1].round().flatten().tolist()
                print target.flatten().tolist()
                mape = mape.new_zeros(1,targetMean.size(0))
                R2Numer = mape.new_zeros(1,targetMean.size(0))
                R2Denom = mape.new_zeros(1,targetMean.size(0))
                loss = []
                cnt = 0
            # perform validation and annealing
            if (batchNum+1) % args.validEvery == 0:
                valid_and_anneal(valSource) 
                model_.train()
                if len(args.abstractDir) != 0:
                    encoder.train()
            # pass over samples that has less than 5 citations till 2005
            if not args.keepLessThan5 and features[0][-1] < 5 : continue 
            if args.embedCategory: # embed citation counts till 2005
                category = torch.LongTensor([util.cnt2category(features[0][-1])]).to(device)

            f_ = torch.FloatTensor(features[0]+features[1]).unsqueeze(0).unsqueeze(2).to(device)
            target = f_.squeeze(2)[:,1:]
            if len(args.abstractDir) != 0:
                abstract = features[2].text.t().to(device)
                abstractTensor = encoder(abstract, hiddenForAbstractEncoder)
            out = model_(f_,hidden, category = None if not args.embedCategory else category, documentVector = None if len(args.abstractDir) == 0 else abstractTensor)
            loss.append(criterion(out[:,:-1],target))

            mape += torch.abs((target[:,len(features[0])-1:]-out[:,len(features[0])-1:-1])/(target[:,len(features[0])-1:]))
            R2Numer += (target[:,len(features[0])-1:]-out[:,len(features[0])-1:-1])**2
            R2Denom += (target[:,len(features[0])-1:]-targetMean)**2
            cnt += 1
            
    except KeyboardInterrupt:
        print 'Exiting from training early...'

def test(dataSource,phase='Validation'):
    model_.eval()
    if len(args.abstractDir) != 0:
        global encoder
        global hiddenForAbstractEncoder
    # if len(trainSource) == 2:
    #     valF, valT = dataSource
    # else:
    #     valF, valT, valAbs = dataSource
    
    # target mean for R2 score
    targetMean=torch.FloatTensor(dataSource[1]).mean(dim=0).to(device)
    mape = torch.zeros(1,targetMean.size(0)).to(device)
    R2Numer = torch.zeros(1,targetMean.size(0)).to(device)
    R2Denom = torch.zeros(1,targetMean.size(0)).to(device)
    cnt = 0
    global hidden
    # eval
    with torch.no_grad():
        for batchNum, features in enumerate(zip(*dataSource)):
            if not args.keepLessThan5 and features[0][-1] < 5 : continue
            if args.embedCategory:
                category = torch.LongTensor([util.cnt2category(features[0][-1])]).to(device)

            f_ = torch.FloatTensor(features[0]).unsqueeze(0).unsqueeze(2).to(device)
            target = torch.FloatTensor(features[1]).unsqueeze(0).to(device)
            if len(args.abstractDir) != 0:
                abstract = features[2].text.t().to(device)
                abstractTensor = encoder(abstract, hiddenForAbstractEncoder)
            for _ in range(len(features[1])):
                out = model_(f_,hidden,category=None if not args.embedCategory else category, documentVector = None if len(args.abstractDir) == 0 else abstractTensor)
                f_ = torch.cat((f_,out.unsqueeze(2)[:,-1:,:]),dim=1)
            if (batchNum+1) % 1000 == 0 :
                print 'pred: %r'%(f_[:,len(features[0]):].round().flatten().tolist())
                print 'answ: %r'%(target.flatten().tolist())

            mape += torch.abs((target-f_.squeeze(2)[:,len(features[0]):])/(target))
            R2Numer += (target-f_.squeeze(2)[:,len(features[0]):])**2
            R2Denom += (target-targetMean)**2
            cnt += 1
        R2 = 1 - R2Numer/R2Denom
        mape = (mape/cnt).flatten().tolist()
        R2 = R2.flatten().tolist()
        print '='*89
        print 'EPOCH %d | %s | cnt : %d | MAPE : %r | R2 : %r'%(EPOCH, phase,cnt,mape, R2)
        print '='*89
    return mape, R2


def valid_and_anneal(dataSource):
    global minValMape
    global lr
    valMape, R2 = test(dataSource)
    evalMetric = sum(valMape) if args.evalMetric == 'mape' else -sum(R2)
    if evalMetric < minValMape:
        minValMape = evalMetric
        print 'writing result and saving model...'
        util.write_result(valMape, R2, args, path=args.resultPath)
        with open(args.save,'wb') as f:
            torch.save((model_.state_dict(),hidden,args),f)
    else:
        lr /= 4.0

if __name__ == '__main__':

    # load data
    print 'Preparing data...'
    lucaData = data.Data(args)

    if len(args.abstractDir) != 0:
        args.vocabDim = len(lucaData.field.vocab.itos)

    # prepare model, optimizer, loss
    if len(args.loadModelPath) != 0:
        model_ = loadModel(args.loadModelPath)
        model_.rnn.flatten_parameters()
    else:
        model_ = model.Many2Many(args)
    
    model_.to(device)
    hidden = model_.init_hidden(device)
    model_.hidden = hidden
    criterion = nn.MSELoss()

    if len(args.abstractDir) != 0:
        # encoder = model.cnnAbs(args)
        encoder = model.lstmAbs(args)
        encoder.to(device)
        hiddenForAbstractEncoder = encoder.init_hidden(device)
    # train
    minValMape = 100000
    lr = args.learningRate

    print 'Training...'
    for epoch in range(args.numEpoch):
        EPOCH = epoch
        train(lucaData.trainSource, lucaData.valSource, args)
    test(lucaData.testSource,phase = 'Test')

