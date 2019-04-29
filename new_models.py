import torch
import torch.nn as nn
import setting
import score
import numpy as np

def wrap(list_of_list, type_=torch.FloatTensor):
    return type_(list_of_list).to(setting.DEVICE)

class Many2One(nn.Module):
    def __init__(self):
        super(Many2One, self).__init__()
        # reg_dim = sum([setting.NUM_FEATURES[k] for k in setting.DATA_TO_USE[setting.DATA_SOURCE]]) - 10
        reg_dim = 70
        # self.paper_embedding_fc = nn.Sequential(nn.Linear(setting.EMBEDDING_DIM,7),nn.ReLU())
        self.scorer = score.Scorer()
        self.feature_types = setting.DATA_TO_USE[setting.DATA_SOURCE]
        self.rnn = nn.GRU(1,70,batch_first=True)
        self.fc = nn.Sequential(nn.Linear(reg_dim, 70), nn.ReLU(), nn.Linear(70,10))
        self.hidden = nn.Parameter(torch.zeros(1,setting.BATCH_SIZE, 70)).to(setting.DEVICE)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=setting.LEARNING_RATE)

    def forward(self, history, paper=None, author=None, venue=None, paper_embedding=None):
        out, _ = self.rnn(history, self.hidden)
        out = out[:,-1,:]
        # feature = torch.cat([paper,author],dim=1)
        # if paper is not None:
        #     out = torch.cat([out, paper], dim=1)
        # if author is not None:
        #     out = torch.cat([out, author], dim=1) 
        # if venue is not None:
        #     out = torch.cat([out, venue], dim=1)
        # if paper_embedding is not None:
        #     paper_embedding = self.paper_embedding_fc(paper_embedding)
        #     out = torch.cat([out, paper_embedding], dim=1)
        pred = self.fc(out)
        return pred

    def fit(self, batch):
        self.train()
        pred = self.forward(wrap(batch['history']).unsqueeze(-1),
                    paper = wrap(batch['paper']) if 'paper' in self.feature_types else None, 
                    author = wrap(batch['author']) if 'author' in self.feature_types else None, 
                    venue = wrap(batch['venue']) if 'venue' in self.feature_types else None, 
                    paper_embedding = wrap(batch['paper_embedding']) if 'paper_embedding' in self.feature_types else None)
        self.scorer.update(pred,wrap(batch['response']))
        self.optimizer.zero_grad()
        self.criterion(pred,wrap(batch['response'])).backward()
        self.optimizer.step()

    def validate(self, data_loader):
        self.eval()
        self.scorer.reset()
        with torch.no_grad():
            for batch in data_loader:
                pred = self.forward(wrap(batch['history']).unsqueeze(-1),
                        paper = wrap(batch['paper']) if 'paper' in self.feature_types else None, 
                        author = wrap(batch['author']) if 'author' in self.feature_types else None, 
                        venue = wrap(batch['venue']) if 'venue' in self.feature_types else None, 
                        paper_embedding = wrap(batch['paper_embedding']) if 'paper_embedding' in self.feature_types else None)
                self.scorer.update(pred, wrap(batch['response']))
            mape, r2 = self.scorer.compute_score()
        return mape, r2

class Many2Many(nn.Module):
    def __init__(self):
        super(Many2Many, self).__init__()
        reg_dim = sum([setting.NUM_FEATURES[k] for k in setting.DATA_TO_USE[setting.DATA_SOURCE].split('_')]) - 10
        self.scorer = score.Scorer()
        self.feature_types = setting.DATA_TO_USE[setting.DATA_SOURCE].split('_')
        self.rnn = nn.GRU(1,2,batch_first=True)
        self.fc = nn.Sequential(nn.Linear(reg_dim, 50), nn.ReLU(), nn.Linear(50,1))
        self.hidden = nn.Parameter(torch.zeros(1,setting.BATCH_SIZE, 2)).to(setting.DEVICE)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=setting.LEARNING_RATE)

    def forward(self, history, response = None, paper=None, author=None, venue=None):
        if response is not None:
            input = torch.cat([history, response],dim=1)
        else: input = history
        out, _ = self.rnn(input, self.hidden)
        if paper is not None:
            out = torch.cat([out, paper.unsqueeze(1).expand(-1,out.size(1),-1)], dim=2)
        if author is not None:
            out = torch.cat([out, author.unsqueeze(1).expand(-1,out.size(1),-1)], dim=2)
        if venue is not None:
            out = torch.cat([out, venue.unsqueeze(1).expand(-1,out.size(1),-1)], dim=2)
        pred = self.fc(out).squeeze(2)[:,:-1]
        return (pred) # +input.squeeze(2)[:,1:]

    def inference(self, history, paper = None, author = None, venue = None):
        for _ in range(10):
            pred = self.forward(history, 
                paper = paper, 
                author = author, 
                venue = venue)
            history = torch.cat([history,pred[:,-1:].unsqueeze(2)],dim=1)
        return history[:,-10:].squeeze(2)

    def fit(self, batch):
        self.train()
        pred = self.forward(wrap(batch['history']).unsqueeze(-1),
                    response = wrap(batch['response']).unsqueeze(-1),
                    paper = wrap(batch['paper']) if 'paper' in self.feature_types else None, 
                    author = wrap(batch['author']) if 'author' in self.feature_types else None, 
                    venue = wrap(batch['venue']) if 'venue' in self.feature_types else None)
        self.scorer.update(pred,wrap(batch['response']))
        self.optimizer.zero_grad()
        target = torch.cat([wrap(batch['history']),wrap(batch['response'])],dim=1)[:,1:]
        self.criterion(pred,target).backward()
        self.optimizer.step()

    def validate(self, data_loader):
        self.eval()
        self.scorer.reset()
        with torch.no_grad():
            for batch in data_loader:
                pred = self.inference(wrap(batch['history']).unsqueeze(-1),
                            paper = wrap(batch['paper']) if 'paper' in self.feature_types else None, 
                            author = wrap(batch['author']) if 'author' in self.feature_types else None, 
                            venue = wrap(batch['venue']) if 'venue' in self.feature_types else None)
                self.scorer.update(pred, wrap(batch['response']))
            mape, r2 = self.scorer.compute_score()
        return mape, r2

class SingleHeadAttention(nn.Module):
    @staticmethod
    def ScaledDotProductAttention(K,Q,V):
        weightLogit = torch.matmul(K,Q.permute([0,2,1])) / np.sqrt(K.size(2)) 
        weight = nn.functional.softmax(weightLogit,dim=1)
        return torch.matmul(weight, V)

    def __init__(self):
        # self.num_heads = setting.NUM_HEADS
        super(SingleHeadAttention,self).__init__()
        self.historyProjector = nn.Sequential(nn.Linear(2,2),nn.ReLU())
        self.authorProjector = nn.Sequential(nn.Linear(5,2),nn.ReLU())
        self.paperProjector = nn.Sequential(nn.Linear(5,2),nn.ReLU())
        self.attention = SingleHeadAttention.ScaledDotProductAttention
        self.layerNorm = nn.modules.normalization.LayerNorm((3,2))
        self.fc = nn.Sequential(nn.Linear(6,6),nn.ReLU())

    def forward(self, history, paper, author, venue=None):
        history = self.historyProjector(history)
        paper = self.paperProjector(paper)
        author = self.authorProjector(author)
        feature = torch.cat([history.unsqueeze(1),paper.unsqueeze(1),author.unsqueeze(1)],dim=1)
        attended_feature = self.attention(feature,feature,feature)
        feature = self.layerNorm(attended_feature).view(setting.BATCH_SIZE,-1)
        feature = self.fc(feature)
        return feature

class Many2OneAttention(nn.Module):
    def __init__(self):
        super(Many2OneAttention, self).__init__()
        self.scorer = score.Scorer()
        self.feature_types = setting.DATA_TO_USE[setting.DATA_SOURCE]
        self.rnn = nn.GRU(1,2,batch_first=True)
        self.fc = nn.Sequential(nn.Linear(18, 50), nn.ReLU(), nn.Linear(50,10))
        self.attention = SingleHeadAttention()
        self.hidden = nn.Parameter(torch.zeros(1,setting.BATCH_SIZE, 2)).to(setting.DEVICE)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=setting.LEARNING_RATE)

    def forward(self, history, paper, author, venue=None):
        out, _ = self.rnn(history, self.hidden)
        out = out[:,-1,:]
        original_feature = torch.cat([out,paper,author],dim=1)
        attended_feature = self.attention(out, paper, author)
        feature = torch.cat([original_feature,attended_feature],dim=1)
        pred = self.fc(feature)
        return pred

    def fit(self, batch):
        self.train()
        pred = self.forward(wrap(batch['history']).unsqueeze(-1),
                    paper = wrap(batch['paper']) if 'paper' in self.feature_types else None, 
                    author = wrap(batch['author']) if 'author' in self.feature_types else None, 
                    venue = wrap(batch['venue']) if 'venue' in self.feature_types else None)
        self.scorer.update(pred,wrap(batch['response']))
        self.optimizer.zero_grad()
        self.criterion(pred,wrap(batch['response'])).backward()
        self.optimizer.step()

    def validate(self, data_loader):
        self.eval()
        self.scorer.reset()
        with torch.no_grad():
            for batch in data_loader:
                pred = self.forward(wrap(batch['history']).unsqueeze(-1),
                        paper = wrap(batch['paper']) if 'paper' in self.feature_types else None, 
                        author = wrap(batch['author']) if 'author' in self.feature_types else None, 
                        venue = wrap(batch['venue']) if 'venue' in self.feature_types else None)
                self.scorer.update(pred, wrap(batch['response']))
            mape, r2 = self.scorer.compute_score()
        return mape, r2