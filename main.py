import math
import torch
import torch.nn as nn
from torchtext import data
from configparser import ConfigParser
from argparse import ArgumentParser

import data
import model
import util

argparser = ArgumentParser()
argparser.add_argument('--numEpoch',type=int, default=10)
argparser.add_argument('--numLayer',type=int, default=1)
argparser.add_argument('--logEvery',type=int, default =2000)
argparser.add_argument('--learningRate',type=float, default = 0.1)
argparser.add_argument('--gradientClip',type=float, default = 0.25)
argparser.add_argument('--seed',type=int, default = 20190102)
argparser.add_argument('--validEvery',type=int, default = 200000, help='perform validation and anneal learning rate every this iterations')
argparser.add_argument('--testRatio',type=float, default = 0.15)
argparser.add_argument('--paperHistoryPath', type=str, default='/home/daehan/Notebooks/citation_prediction/data/lucaweihs/paperHistories-1975-2005-2015-2.tsv')
argparser.add_argument('--loadModelPath', type=str, default='', help = 'path to checkpoint to resume training')
argparser.add_argument('--resultPath', type=str, default='result.txt', help='name of result file')
argparser.add_argument('--save', type=str, default='model.pt', help = 'path to checkpoint to save model')
argparser.add_argument('--paperResponsePath', type=str, default='/home/daehan/Notebooks/citation_prediction/data/lucaweihs/paperResponses-1975-2005-2015-2.tsv')
argparser.add_argument('--cuda',action='store_true')
argparser.add_argument('--bidir',action='store_true')
argparser.add_argument('--embedCategory',action='store_true', help='Whether to use category embedding')
argparser.add_argument('--debug',action='store_true', help='To test a functionality, use this flag for shorter running time (with --reprocess flag)')

argparser.add_argument('--reprocess',action='store_true',help='process whole data again')

args = argparser.parse_args()

device = torch.device('cuda' if args.cuda else 'cpu')


def train(trainSource,valSource,args):
    #data load
    global model_
    global hidden 
    global lr
    trainF,trainT = trainSource
    # target mean for R2 score
    targetMean=torch.FloatTensor(trainT).mean(dim=0).to(device)
    mape = torch.zeros(1,10).to(device)
    R2Numer = torch.zeros(1,10).to(device)
    R2Denom = torch.zeros(1,10).to(device)
    loss = []
    cnt = 0
    try:
        for batchNum, (feature1, feature2) in enumerate(zip(trainF,trainT)):
            if (batchNum+1) % args.logEvery == 0:
                (sum(loss)/cnt).backward()
                torch.nn.utils.clip_grad_norm_(model_.parameters(), args.gradientClip)
                torch.nn.utils.clip_grad_norm_(hidden, args.gradientClip)
                for p in model_.parameters(): # learning model parameters
                    p.data.add_(-lr, p.grad)
                for h in hidden: # learning initial hidden state
                    h.data.add_(-lr, h.grad)
                    h.grad = torch.zeros_like(h)
                model_.zero_grad()
                # hidden = util.repackage_hidden(hidden)
                R2 = 1 - R2Numer/R2Denom
                print 'batch %d/%d | lr : %.6f | cnt : %d | train MAPE : %r | train R2 : %r'%(batchNum+1,len(trainF), lr ,cnt,(mape/cnt).flatten().tolist(), R2.flatten().tolist())
                mape = mape.new_zeros(1,10)
                R2Numer = mape.new_zeros(1,10)
                R2Denom = mape.new_zeros(1,10)
                loss = []
                cnt = 0
            # perform validation and annealing
            if (batchNum+1) % args.validEvery == 0:
                valid_and_anneal(valSource) 
                model_.train()
            # pass over samples that has less than 5 citations till 2005
            if feature1[-1] < 5 : continue 
            if args.embedCategory: # embed citation counts till 2005
                category = torch.LongTensor([util.cnt2category(feature1[-1])]).to(device)

            f_ = torch.FloatTensor(feature1+feature2).unsqueeze(0).unsqueeze(2).to(device)
            target = f_.squeeze(2)[:,1:]
            out = model_(f_,hidden, category = None if not args.embedCategory else category)
            loss.append(criterion(out[:,:-1],target))
            mape += torch.abs((target[:,len(feature1)-1:]-out[:,len(feature1)-1:-1])/(target[:,len(feature1)-1:]))
            R2Numer += (target[:,len(feature1)-1:]-out[:,len(feature1)-1:-1])**2
            R2Denom += (target[:,len(feature1)-1:]-targetMean)**2
            cnt += 1
            
    except KeyboardInterrupt:
        print 'Exiting from training early...'

def test(dataSource,phase='Validation'):
    model_.eval()
    valF, valT = dataSource
    # target mean for R2 score
    targetMean=torch.FloatTensor(valT).mean(dim=0).to(device)
    mape = torch.zeros(1,10).to(device)
    R2Numer = torch.zeros(1,10).to(device)
    R2Denom = torch.zeros(1,10).to(device)
    cnt = 0
    global hidden
    # eval
    with torch.no_grad():
        for batchNum, (feature1, feature2) in enumerate(zip(valF,valT)):
            if feature1[-1] < 5 : continue
            if args.embedCategory:
                category = torch.LongTensor([util.cnt2category(feature1[-1])]).to(device)
            f_ = torch.FloatTensor(feature1).unsqueeze(0).unsqueeze(2).to(device)
            target = torch.FloatTensor(feature2).unsqueeze(0).to(device)
            for _ in range(len(feature2)):
                out = model_(f_,hidden,category=None if not args.embedCategory else category)
                f_ = torch.cat((f_,out.unsqueeze(2)[:,-1:,:]),dim=1)
            mape += torch.abs((target-f_.squeeze(2)[:,len(feature1):])/(target))
            R2Numer += (target-f_.squeeze(2)[:,len(feature1):])**2
            R2Denom += (target-targetMean)**2
            cnt += 1
        R2 = 1 - R2Numer/R2Denom
        mape = (mape/cnt).flatten().tolist()
        R2 = R2.flatten().tolist()
        print '='*89
        print '%s | cnt : %d | MAPE : %r | R2 : %r'%(phase,cnt,mape, R2)
        print '='*89
    return mape, R2

def valid_and_anneal(dataSource):
    global minValMape
    global lr
    valMape, R2 = test(dataSource)
    if sum(valMape) < minValMape:
        minValMape = sum(valMape)
        print 'writing result and saving model...'
        util.write_result(valMape, R2,path=args.resultPath)
        with open(args.save,'wb') as f:
            torch.save((model_.state_dict(),hidden,args),f)
    else:
        lr /= 4.0

if __name__ == '__main__':

    # load data
    print 'Preparing data...'
    lucaData = data.Data(args)

    # prepare model, optimizer, loss
    if len(args.loadModelPath) != 0:
        model_ = loadModel(args.loadModelPath)
        model_.rnn.flatten_parameters()
    else:
        model_ = model.basicLSTM(args)
    model_.to(device)
    hidden = model_.init_hidden(device)
    model_.hidden = hidden
    criterion = nn.MSELoss()

    # train
    minValMape = 10
    lr = args.learningRate

    print 'Training...'
    for epoch in range(args.numEpoch):
        train((lucaData.train,lucaData.trainTarget), (lucaData.val,lucaData.valTarget),args)
    test((lucaData.test,lucaData.testTarget),phase = 'Test')

