import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import numpy as np
import setting
from sklearn.metrics import r2_score

def mean_absolute_percentage_error(pred,target):
    return np.absolute((np.array(target) - np.array(pred))/ np.array(target)).mean()

class plotter(object):
    def __init__(self):
        self.path = '{}_fig_last_year.png'
        self.pred_list = [] 
        self.target_list = []

    def reset(self):
        self.pred_list = []
        self.target_list = []

    def update(self, pred, target, history):
        # pred, out : torch.Tensor / batch_number x prediction_years
        history_sum = history.squeeze(2).sum(1)
        
        cum_pred = pred.cumsum(dim=1) + history_sum.unsqueeze(1)
        cum_target = target.cumsum(dim=1) + history_sum.unsqueeze(1)

        self.pred_list += cum_pred[:,-1].tolist()
        self.target_list += cum_target[:,-1].tolist()

    def compute_scores(self):
        self.R2 = r2_score(self.pred_list,self.target_list)
        self.mape = mean_absolute_percentage_error(self.pred_list, self.target_list)

    def plot(self):
        self.compute_scores()
        t = np.arange(0,max(self.target_list),0.01)
        fig = plt.figure(1)
        ax = plt.subplot(111)
        ax.title.set_text('R2 : {:.2f} / MAPE : {:.2f}'.format(self.R2,self.mape))
        ax.set_xlabel('pred')
        ax.set_ylabel('target')
        ax.plot(t,t,'k',self.pred_list, self.target_list,'bo')
        plt.savefig(self.path.format(setting.DATA_SOURCE),format='png')
        plt.clf()

