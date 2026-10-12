import copy
import random
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix

class MeanSAGE(nn.Module):
    """Two-layer mean GraphSAGE using native PyTorch sparse adjacency.

    h' = W_self h + W_neighbour mean(incoming h). No global normalization
    or batch statistics: future/held-out nodes cannot influence disconnected nodes.
    """
    def __init__(self, inputs, hidden=32, neighbor=True):
        super().__init__()
        self.neighbor=neighbor
        self.self_layers=nn.ModuleList([nn.Linear(inputs,hidden),nn.Linear(hidden,hidden)])
        self.neighbor_layers=nn.ModuleList([nn.Linear(inputs,hidden,bias=False),nn.Linear(hidden,hidden,bias=False)])
        if not neighbor:
            for parameter in self.neighbor_layers.parameters(): parameter.requires_grad_(False)
        self.head=nn.Linear(hidden,2)

    def forward(self,x,adj):
        for root,near in zip(self.self_layers,self.neighbor_layers):
            x=torch.relu(root(x)+(near(torch.sparse.mm(adj,x)) if self.neighbor else 0))
        return self.head(x)

class MLP(nn.Module):
    def __init__(self,inputs,hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(inputs,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,2))
    def forward(self,x,adj=None):return self.net(x)

def counts(model):
    return {'total':sum(p.numel() for p in model.parameters()),
            'active':sum(p.numel() for p in model.parameters() if p.requires_grad)}

def matched_mlp_width(inputs,sage_hidden):
    target=counts(MeanSAGE(inputs,sage_hidden))['active']
    return min(range(1,4*sage_hidden+1), key=lambda h:abs((inputs*h+h+h*h+h+2*h+2)-target))

def adjacency(edges,n):
    if len(edges):
        degree=np.bincount(edges[:,1],minlength=n)
        indices=torch.tensor(edges[:,::-1].copy().T,dtype=torch.long)
        values=torch.tensor(1.0/degree[edges[:,1]],dtype=torch.float32)
    else:
        indices=torch.empty((2,0),dtype=torch.long);values=torch.empty(0)
    return torch.sparse_coo_tensor(indices,values,(n,n),check_invariants=True).coalesce()

def train_neural(x,y,splits,train_edges,val_edges,seed,kind,hidden=32,epochs=100,patience=15):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    width=matched_mlp_width(x.shape[1],hidden)
    model=MLP(x.shape[1],width) if kind=='mlp' else MeanSAGE(x.shape[1],hidden,kind!='sage_no_neighbors')
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-4)
    xt=torch.tensor(x);yt=torch.tensor(y,dtype=torch.long)
    tr=torch.tensor(splits=='train');va=torch.tensor(splits=='val')
    freq=np.bincount(y[splits=='train'],minlength=2)
    weight=torch.tensor(freq.sum()/(2*freq),dtype=torch.float32)
    lossfn=nn.CrossEntropyLoss(weight=weight)
    at=adjacency(train_edges,len(x));av=adjacency(val_edges,len(x))
    best_loss=float('inf');best=None;stale=0;history=[]
    for epoch in range(epochs):
        model.train();opt.zero_grad()
        logits=model(xt,at);loss=lossfn(logits[tr],yt[tr]);loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): val=float(lossfn(model(xt,av)[va],yt[va]))
        history.append({'epoch':epoch+1,'train_loss':float(loss.detach()),'val_loss':val})
        if val<best_loss-1e-6:
            best_loss=val;best=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best)
    return model,{'parameter_counts':counts(model),'mlp_width':width if kind=='mlp' else None,
                  'best_validation_loss':best_loss,'history':history}

def predict(model,x,edges):
    model.eval()
    with torch.no_grad():return torch.softmax(model(torch.tensor(x),adjacency(edges,len(x))),dim=1)[:,1].numpy()

def train_rf(x,y,splits,seed):
    m=RandomForestClassifier(n_estimators=200,class_weight='balanced',n_jobs=2,random_state=seed)
    m.fit(x[splits=='train'],y[splits=='train']);return m

def metrics(y,prob):
    prediction=(prob>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
    return {'macro_f1':float(f1_score(y,prediction,labels=[0,1],average='macro',zero_division=0)),
            'attack_recall':float(tp/(tp+fn)) if tp+fn else None,
            'benign_false_positive_rate':float(fp/(tn+fp)) if tn+fp else None,
            'average_precision':float(average_precision_score(y,prob)) if len(np.unique(y))==2 else None,
            'support_benign':int(tn+fp),'support_attack':int(tp+fn),
            'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]], 'threshold':.5}
