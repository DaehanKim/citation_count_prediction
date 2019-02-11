import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def scaledDotProductAttention(K,Q,V):
    # K : 1,100 
    # Q : 1,seq_len,100
    # V : 1,seq_len,100
    alpha = F.softmax(torch.matmul(Q,K.t())/math.sqrt(K.size(1)), dim=1) #1,seq_len,1
    return alpha*V


class Many2Many(nn.Module): 
    def __init__(self,args):
        super(Many2Many, self).__init__()
        self.args = args
        self.rnn = getattr(nn,args.modelType)(1,100,args.numLayer,batch_first=True) # dim_input, dim_hidden, num_layer
        if args.embedCategory:
            self.emb = nn.Embedding(20, 100)
            self.drop = nn.Dropout(p=0.2)
        if args.embedCategory and not args.attention:
            regressorDim = 300 if len(args.abstractDir) != 0 else 200 
        else:
            regressorDim = 200 if len(args.abstractDir) != 0 else 100
        self.regressor = nn.Sequential(nn.Linear(regressorDim, 10), nn.ReLU() ,nn.Linear(10,1))
    
    def forward(self, x, hidden, category=None, documentVector=None):
        # print x
        output, _ = self.rnn(x,hidden)
        if self.args.embedCategory: 
            emb = self.drop(self.emb(category))
        if self.args.attention:
            attValues = []
            for i in range(1,output.size(1)+1):
                attValues.append(scaledDotProductAttention(emb, output[:,:i,:], output[:,:i,:]).sum(dim=1).unsqueeze(1)) # batch_size, sequenceLen, hidden_dim 
            output = torch.cat(attValues,dim=1) # 1,1,hidden_dim
        elif self.args.embedCategory:
            emb = torch.cat([emb.unsqueeze(1) for _ in range(output.size(1))], dim=1)
            output = torch.cat([output, emb],dim=2)
        if documentVector is not None:
            # 1,doc_dim
            output = torch.cat([output,documentVector.repeat(1,x.size(1),1)], dim = 2)
        output = self.regressor(output)

        return (output+x).squeeze(2)

    def init_hidden(self,device,bsz=1):
        if self.args.modelType == 'LSTM':
            return tuple(nn.Parameter(torch.zeros((self.args.numLayer,bsz, 100),device=device)) for _ in range(2))
        else:
            return nn.Parameter(torch.zeros((self.args.numLayer,bsz, 100),device=device))

class Many2ManyEncoderDecoder(nn.Module): 
    def __init__(self,args):
        super(Many2ManyEncoderDecoder, self).__init__()
        self.args = args
        self.rnnEncoder = getattr(nn,args.modelType)(1,100,args.numLayer,batch_first=True) # dim_input, dim_hidden, num_layer

        if args.embedCategory:
            self.emb = nn.Embedding(20, 100)
            self.drop = nn.Dropout(p=0.2)
        if args.embedCategory and not args.attention:
            regressorDim = 300 if len(args.abstractDir) != 0 else 200 
        else:
            regressorDim = 200 if len(args.abstractDir) != 0 else 100
        self.regressorDim = regressorDim
        self.rnnDecoder = getattr(nn,args.modelType)(regressorDim,1,args.numLayer,batch_first=True) # dim_input, dim_hidden, num_layer

    def forward(self, x, hidden1, hidden2, category=None, documentVector=None):

        output, _ = self.rnnEncoder(x,hidden1)
        output = F.relu(output)
        if self.args.embedCategory: 
            emb = self.drop(self.emb(category))
        if self.args.attention:
            attValues = []
            for i in range(1,output.size(1)+1):
                attValues.append(scaledDotProductAttention(emb, output[:,:i,:], output[:,:i,:]).sum(dim=1).unsqueeze(1)) # batch_size, sequenceLen, hidden_dim 
            output = torch.cat(attValues,dim=1) # 1,1,hidden_dim
        elif self.args.embedCategory:
            emb = torch.cat([emb.unsqueeze(1) for _ in range(output.size(1))], dim=1)
            output = torch.cat([output, emb],dim=2)
        if documentVector is not None:
            # 1,doc_dim
            output = torch.cat([output,documentVector.repeat(1,x.size(1),1)], dim = 2)
        output, _ = self.rnnDecoder(output, hidden2)

        return (output+x).squeeze(2)
    
    def init_hidden(self,device,bsz=1):
        if self.args.modelType == 'LSTM':
            return tuple(nn.Parameter(torch.zeros((self.args.numLayer,bsz, 100),device=device)) for _ in range(2)), tuple(nn.Parameter(torch.zeros((self.args.numLayer,bsz, 1),device=device)) for _ in range(2)) 
        else:
            return nn.Parameter(torch.zeros((self.args.numLayer,bsz, 100),device=device)), nn.Parameter(torch.zeros((self.args.numLayer,bsz, 1),device=device))


class Many2One(nn.Module): 
    def __init__(self,args):
        super(Many2One, self).__init__()
        self.args = args
        self.rnn = getattr(nn,args.modelType)(1,100,args.numLayer,batch_first=True) # dim_input, dim_hidden, num_layer
        if args.embedCategory:
            self.emb = nn.Embedding(20, 100)
            self.drop = nn.Dropout(p=0.2)
        self.regressor = nn.Sequential(nn.Linear(200 if args.embedCategory and not args.attention else 100, 10), nn.ReLU() ,nn.Linear(10,10))
    
    def forward(self, x, hidden, category=None):

        output, _ = self.rnn(x,hidden)
        if self.args.embedCategory: 
            emb = self.drop(self.emb(category))
        if self.args.attention:
            output = scaledDotProductAttention(emb, output, output).sum(dim=1).unsqueeze(1) # batch_size, sequenceLen, hidden_dim 
            output = output.sum(dim=1)# 1,hidden_dim
        if output.dim() == 3:
            output = output[:,-1,:]
        if self.args.embedCategory and not self.args.attention:
            print output.size(), emb.size()
            output = torch.cat([output, emb],dim=1)
        output = self.regressor(output)
        return torch.cat([output[:,:1],output[:,1:] + output[:,:-1]],dim=1).contiguous()
    
    def init_hidden(self,device,bsz=1):
        if self.args.modelType == 'LSTM':
            return tuple(nn.Parameter(torch.zeros((self.args.numLayer,bsz, 100),device=device)) for _ in range(2))
        else:
            return nn.Parameter(torch.zeros((self.args.numLayer,bsz, 100),device=device))

class cnnAbs(nn.Module):
    def __init__(self,args):
        super(cnnAbs, self).__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocabDim,100)
        self.drop = nn.Dropout(p=0.2)
        self.convs = nn.ModuleList([nn.Conv2d(1,50,(i,100)) for i in range(3,6)])
        self.regressor = nn.Sequential(nn.Linear(150,100), nn.ReLU(), nn.Linear(100,100))

    def forward(self, x):
        # x : batch_size, seq_len
        out = self.drop(self.emb(x)).unsqueeze(1) # out : batch_size, seq_len, word_dim
        # print out.size()
        outs = [F.relu(conv(out).max(dim=2)[0]) for conv in self.convs]
        out = torch.cat(outs,dim=1).squeeze(2)
        out = self.regressor(out)
        return out # [1, 100]

class lstmAbs(nn.Module):
    def __init__(self,args):
        super(lstmAbs, self).__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocabDim,100)
        self.drop = nn.Dropout(p=0.2)
        self.rnn = getattr(nn,args.encoderType)(100,200,args.numLayer,batch_first=True, bidirectional=args.bidir) # dim_input, dim_hidden, num_layer
        self.regressor = nn.Sequential(nn.Linear(200+int(args.bidir)*200,100), nn.ReLU(), nn.Linear(100,100))

    def forward(self, x, hidden):
        # x : batch_size, seq_len
        out = self.drop(self.emb(x)) # out : batch_size, seq_len, word_dim
        out, _ = self.rnn(out, hidden) # out : batch, seq_len, hidden_vector, 1, 154, 400
        out = self.regressor(out) # out: 1, 154, 100
        return out[:,-1,:].squeeze(1)

    def init_hidden(self,device,bsz=1):
        if self.args.encoderType == 'LSTM':
            return tuple(nn.Parameter(torch.zeros((self.args.numLayer*(int(self.args.bidir)+1),bsz, 200),device=device)) for _ in range(2))
        else:
            return nn.Parameter(torch.zeros((self.args.numLayer*(int(self.args.bidir)+1),bsz, 200),device=device))
