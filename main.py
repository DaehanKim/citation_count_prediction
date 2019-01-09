import math
import torch
import torch.nn as nn
from torchtext import data
from configparser import ConfigParser
from argparse import ArgumentParser

import data
import model 

argparser = ArgumentParser()
argparser.add_argument('--numEpoch',type=int, default=10)
argparser.add_argument('--logEvery',type=int, default =2000)
argparser.add_argument('--learningRate',type=float, default = 0.1)
argparser.add_argument('--gradientClip',type=float, default = 0.25)
argparser.add_argument('--cuda',action='store_true')
argparser.add_argument('--samplePrediction',action='store_true')
argparser.add_argument('--debug',action='store_true')
argparser.add_argument('--reprocess',action='store_true',help='process whole data again')

args = argparser.parse_args()

config = ConfigParser()
config.read('config.ini')
config.set('general','debug','True' if args.debug else 'False')
config.set('general','reprocess','True' if args.reprocess else 'False')

device = torch.device('cuda' if args.cuda else 'cpu')


def train(dataSource,args):
    #data load
    trainF,trainT = dataSource
    # target mean for R2 score
    targetMean=torch.FloatTensor(trainT).mean(dim=0)
    mape = torch.zeros(1,10)
    R2Numer = torch.zeros(1,10)
    R2Denom = torch.zeros(1,10)
    loss = []
    cnt = 0
    try:
        for batchNum, (feature1, feature2) in enumerate(zip(trainF,trainT)):
            if (batchNum+1) % args.logEvery == 0:
                (sum(loss)/cnt).backward()
                torch.nn.utils.clip_grad_norm_(model_.parameters(), args.gradientClip)
                for p in model_.parameters():
                    p.data.add_(-lr, p.grad.data)

                R2 = 1 - R2Numer/R2Denom
                print 'batch %d/%d | cnt : %d | train MAPE : %r | train R2 : %r'%(batchNum+1,len(trainF), cnt,(mape/cnt).flatten().tolist(), R2.flatten().tolist())
                if args.samplePrediction:
                    print 'prediction : %r'%(out[:,len(feature1)-1:-1].round().flatten().tolist())
                    print 'answer : %r'%(target[:,len(feature1)-1:].flatten().tolist())
                mape = torch.zeros(1,10)
                R2Numer = torch.zeros(1,10)
                R2Denom = torch.zeros(1,10)
                loss = []
                cnt = 0
            if feature1[-1] < 5 : continue
            f_ = torch.FloatTensor(feature1+feature2).unsqueeze(0).unsqueeze(2).to(device)
            target = f_.squeeze(2)[:,1:]
            out = model_(f_,hidden)
            loss.append(criterion(out[:,:-1],target))
            mape += torch.abs((target[:,len(feature1)-1:]-out[:,len(feature1)-1:-1])/(target[:,len(feature1)-1:]))
            R2Numer += (target[:,len(feature1)-1:]-out[:,len(feature1)-1:-1])**2
            R2Denom += (target[:,len(feature1)-1:]-targetMean)**2
            cnt += 1
            
    except KeyboardInterrupt:
        print 'saving model...'
        with open('./model.pt','wb') as f:
            torch.save((model_.state_dict(),hidden),f)


def test(dataSource):
    model_.eval()
    valF, valT = dataSource
    # target mean for R2 score
    targetMean=torch.FloatTensor(valT).mean(dim=0)
    mape = torch.zeros(1,10)
    R2Numer = torch.zeros(1,10)
    R2Denom = torch.zeros(1,10)
    cnt = 0
    # eval
    with torch.no_grad():
        for batchNum, (feature1, feature2) in enumerate(zip(valF,valT)):
            if (batchNum+1) % args.logEvery == 0 and args.samplePrediction:
                print 'prediction : %r'%(f_.squeeze(2)[:,len(feature1):].round().flatten().tolist())
                print 'answer : %r'%(target.flatten().tolist())
            if feature1[-1] < 5 : continue
            f_ = torch.FloatTensor(feature1).unsqueeze(0).unsqueeze(2).to(device)
            target = torch.FloatTensor(feature2).unsqueeze(0).to(device)
            for _ in range(len(feature2)):
                out = model_(f_,hidden)
                f_ = torch.cat((f_,out.unsqueeze(2)[:,-1:,:]),dim=1)
            mape += torch.abs((target-f_.squeeze(2)[:,len(feature1):])/(target))
            R2Numer += (target-f_.squeeze(2)[:,len(feature1):])**2
            R2Denom += (target-targetMean)**2
            cnt += 1
            

        R2 = 1 - R2Numer/R2Denom
        print 'Validation | cnt : %d | MAPE : %r | R2 : %r'%(cnt,(mape/cnt).flatten().tolist(), R2.flatten().tolist())
        mape = torch.zeros(1,10)
        R2Numer = torch.zeros(1,10)
        R2Denom = torch.zeros(1,10)
        cnt = 0
    return mape.sum()


# load data
print 'Preparing data...'
lucaData = data.Data(config)

# prepare model, optimizer, loss
model_ = model.basicLSTM(args)
model_.to(device)
hidden = model_.init_hidden(device)
# optimizer = torch.optim.Adam(list(model_.parameters())+list(hidden),lr=args.learningRate)
criterion = nn.MSELoss()

# train
minValMape = 10
lr = args.learningRate

print 'Training...'
for epoch in range(args.numEpoch):
    train((lucaData.train,lucaData.trainTarget), args)
    valMape = test((lucaData.val, lucaData.valTarget))
    if valMape < minValMape:
        minValMape = valMape
        print 'saving model...'
        with open('./model.pt','wb') as f:
            torch.save((model_.state_dict(),hidden),f)
    else:
        lr /= 5.0