import torch.nn as nn
import torch

class basicLSTM(nn.Module): 
    def __init__(self,args):
        super(basicLSTM, self).__init__()
        self.rnn = nn.LSTM(1,10,1,batch_first=True) # dim_input, dim_hidden, num_layer
        self.regressor = nn.Sequential(nn.Linear(10, 10), nn.ReLU() ,nn.Linear(10,1))
    
    def forward(self, x, hidden):
        output, (_, _) = self.rnn(x,hidden)
        output = self.regressor(output)
        return (output+x).squeeze(2)
    
    def init_hidden(self,device,bsz=1):
        return (torch.zeros((1,bsz, 10),device=device),torch.zeros((1,bsz, 10),device=device))
