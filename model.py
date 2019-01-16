import torch.nn as nn
import torch

class basicLSTM(nn.Module): 
    def __init__(self,args):
        super(basicLSTM, self).__init__()
        self.args = args
        self.rnn = nn.LSTM(1,100,1,batch_first=True) # dim_input, dim_hidden, num_layer
        if args.embedCategory:
            self.emb = nn.Embedding(20, 100)
            self.drop = nn.Dropout(p=0.2)
        self.regressor = nn.Sequential(nn.Linear(200 if args.embedCategory else 100, 10), nn.ReLU() ,nn.Linear(10,1))
    
    def forward(self, x, hidden, category=None):
        output, (_, _) = self.rnn(x,hidden)
        if self.args.embedCategory: 
            emb = self.drop(self.emb(category))
            emb = torch.cat([emb.unsqueeze(1) for _ in range(output.size(1))], dim=1)
            output = torch.cat([output, emb],dim=2)
        output = self.regressor(output)
        return (output+x).squeeze(2)
    
    def init_hidden(self,device,bsz=1):
        return tuple(nn.Parameter(torch.zeros((1,bsz, 100),device=device)) for _ in range(2))