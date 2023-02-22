import numpy as np
import pandas as pd
import pickle
import os, warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('always')	# "error", "ignore", "always", "default", "module" or "once"

lr = 0.001
epochs = 200
batch_size = 2
p_list = [0.1, 0.05, 0.01, 0.005]
PATH = "./data/"


def split_samples(data, path) :
    X = data.iloc[:,1:]
    y = data["label"]
    samples = {}
    for fold in range(10):
        samples[fold] = {}
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = fold, stratify=y)

        samples[fold]["train"] = y_train.index.tolist()
        samples[fold]["test"] = y_test.index.tolist()

    with open(path + f"split_samples.pkl","wb") as f:
        pickle.dump(samples,f)
    return samples

def get_results(labels, predict_proba):
	predict_proba = np.array(predict_proba).reshape(-1,1)
	AUC = roc_auc_score(labels, predict_proba)
	return AUC

class FC_Layer(nn.Module):			
	def __init__(self, n_feature):
		super(FC_Layer, self).__init__()

		self.model = nn.Sequential(
		nn.Linear(n_feature, 128, bias = True), 
		nn.ReLU(),
		nn.Linear(128,	64, bias = True),
		nn.ReLU(),
		nn.Linear(64,	32, bias = True),
		nn.ReLU(),
		nn.Linear(32, 1), # output (binary)
		nn.Sigmoid())
	def forward(self,x):
		return self.model(x)
        
def train(model, train_loader, optimizer, loss_function) :
    model.train()
    train_loss = 0
    
    for idx, (batch_data, batch_labels) in enumerate(train_loader) :
        outputs = model(batch_data)
        loss = loss_function(outputs,batch_labels.view(-1,1))
        
        train_loss += loss.item()
        
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    return train_loss
    
def evaluate(model,test_loader) :
    model.eval()
    batch_labels_list = []
    batch_prediciton_list = []
    
    for idx, (batch_data, batch_labels) in enumerate(test_loader):
        outputs = model(batch_data)
        
        batch_labels_list.extend(batch_labels.detach().numpy())
        batch_prediciton_list.extend(outputs.detach().numpy())
    
    output = get_results(batch_labels_list,batch_prediciton_list)
    return output
    
def calculate_p_value(exp_data, genes, X_train_idx) :
    exp_good = exp_data[exp_data['label']=='0']
    exp_bad = exp_data[exp_data['label']=='1']
    
    idx_good = np.array(set(exp_good.index) & set(X_train_index))
    good = exp_good.loc[idx_good,:]
    idx_bad = np.array(set(exp_bad.index) & set(X_train_index))
    bad = exp_bad.loc[idx_bad,:]
    
    p_values = []
    for i in genes :
        t_v, p_v = ttest_ind(good[i].values, bad[i].values)
        p_values.append(p_v)
        
    p_values = np.array(p_values)
    return p_values

######### data load ##########

#expression
exp_data = pd.read_csv(PATH + "eset_T.csv",index_col=0)

#score
with open(PATH + f"Good_samples.pkl","rb") as f:
    data_x1 = pickle.load(f) # good score samples
with open(PATH + f"Bad_samples.pkl","rb") as f:
    data_x2 = pickle.load(f) # bad score data
genes = data_x1.columns.values.reshape(-1) # gene list

sample_list = split_samples(exp_data, PATH)# sample list dict / for 10 independent test

####################################

data_x1['label'] = 0
data_x2['label'] = 1
data_X = pd.concat([data_x1, data_x2], axis = 0)
y = data_X.iloc[:,-1] # label
X = data_X.drop(['label'],axis=1)

####################################

total_p = {}

for cnt in range(10) :
    # data setting
    X_train = X.loc[sample_list[cnt]["train"]]
    X_test = X.loc[sample_list[cnt]["test"]]
    y_train = y.loc[sample_list[cnt]["train"]]
    y_test = y.loc[sample_list[cnt]["test"]]
    
    find_p = pd.DataFrame()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 115)
    cv = 0
    for train_index, test_index in skf.split(X_train, y_train): # 5 cross validation
        cv_X_train, cv_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        cv_y_train, cv_y_test = y_train.iloc[train_index], y_train.iloc[test_index]

        p_values = calculate_p_value(exp_data, genes, cv_X_train.index)
        result = []

        print(f"=={cnt} - {cv}==")
        cv += 1

        for p in p_list :	
            sel_gene = genes[p_values<=p]
            print(f"{p} - sel_gene : ",len(sel_gene))	
                
            if len(sel_gene) < 10 : # Less than 10 features are too few, so test is omitted.
                result.append([np.NaN])
                continue
            p_X_train = cv_X_train.loc[:,sel_gene]
            p_X_test = cv_X_test.loc[:,sel_gene]
            sel_gene = 0
            tensor_train_X = torch.Tensor(p_X_train.values)
            tensor_train_Y = torch.Tensor(cv_y_train)
            tensor_test_X = torch.Tensor(p_X_test.values)
            tensor_test_Y = torch.Tensor(cv_y_test)
            
            train_data = TensorDataset(tensor_train_X,tensor_train_Y)
            test_data = TensorDataset(tensor_test_X,tensor_test_Y)
            
            clf = FC_Layer(n_feature = tensor_train_X.shape[1]) 
            
            train_loader = DataLoader(train_data, batch_size,shuffle=True)
            test_loader = DataLoader(test_data, batch_size)
            
            opt = Adam(clf.parameters(), lr = lr)
            loss_func = nn.BCELoss(reduction='sum') 
            
            patience = 0
            for epoch in range(epochs):
                train_loss = train(clf,train_loader, opt, loss_func)
                loss = train_loss/len(tensor_train_Y)
                
                if loss < 0.0001 :
                    patience += 1
                    
                if patience > 20 :
                    break
                    
            print(f"epoch : {epoch} - LOSS : {loss}")
            
            output = evaluate(clf,test_loader)
            result.append([output])
        
        result = pd.DataFrame(result,index=["0.1","0.05","0.01","0.005"], columns=["AUC"])
        find_p = pd.concat([find_p,result], axis =1)
        result = 0
    print(pd.concat([find_p, find_p.mean(axis = 1),find_p.std(axis=1)],keys=['','mean','std'] axis =1))
    
    mean_p = find_p.mean(axis = 1).sort_values(ascending = False)
    
    ##########
    
    best_p = float(mean_p.index[0])
    
    p_values = calculate_p_value(exp_data,genes,X_train.index)
    
    sel_gene = genes[p_values<=best_p]
    print(f"p-value : {best_p} / selected genes : ",len(sel_gene))	
    
    X_train = X_train.loc[:,sel_gene]
    X_test = X_test.loc[:,sel_gene]
    train_X = torch.Tensor(X_train.values)
    train_Y = torch.Tensor(y_train)
    test_X = torch.Tensor(X_test.values)
    test_Y = torch.Tensor(y_test)
    
    train_data = TensorDataset(train_X,train_Y)
    test_data = TensorDataset(test_X,test_Y)
    
    clf = FC_Layer(n_feature = train_X.shape[1]) 
    
    train_loader = DataLoader(train_data, batch_size,shuffle=True)
    test_loader = DataLoader(test_data, batch_size)
    
    opt = Adam(clf.parameters(), lr = lr)
    loss_func = nn.BCELoss(reduction='sum') 

    patience = 0
    for epoch in range(epochs):
        train_loss = train(clf,train_loader,opt,loss_func)
        
        loss = train_loss/len(train_Y)
        
        if loss < 0.0001 :
            patience += 1
            
        if patience > 20 :
            break
            
    print(f" - LOSS",epoch,"\t",loss)
    
    output = evaluate(clf,test_loader)
    
    print(f"{cnt} - AUC : {output}")
    total_p[cnt] = [best_p, output]

df = pd.DataFrame(total_p, index = ['p-value','AUC'])
df = pd.concat([df,df.mean(axis = 1),df.std(axis = 1)],key=['','mean','std'],axis = 1).T
print(df)
df.to_csv("DNN_AUC_result.csv")