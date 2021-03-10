# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:51:42 2021

@author: Georgios N. Dimitrakopoulos, geodimitrak@upatras.gr
"""

from xgboost import  XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
import pickle
import random
import time


random.seed(2021)


SV = 0.5 #supervision percentage
P = 1 - SV
SAVE = True

(gex_tf, gex_target, adj, is_diabetes, G1, G2) = pickle.load(open('./data/diabetes.pkl','rb'))

#log transform data
gex_tf = np.log(gex_tf+1)
gex_target = np.log(gex_target+1)

gex_tf = gex_tf[:,is_diabetes==0]
gex_target = gex_target[:,is_diabetes==0]


targets_per_tf = np.sum(adj, axis=1)

net_mse = np.zeros(adj.shape)
net_rsq = np.zeros(adj.shape)
net_mae = np.zeros(adj.shape)

tr = [] #list of targets used in training

start = time.time()
for i in range(len(G1)):
    if(targets_per_tf[i]==0):
        tr.append([]) #no known targets
        continue
    
    #split targets to P% test and rest as train
    tmp = adj[i,:]>0
    targets = np.where(tmp)[0]
    n_test = np.ceil(len(targets) * P).astype('int32')
    random.shuffle(targets)
    targets_test = targets[0:n_test]
    targets_train = targets[n_test:len(targets)]
    
    #fix for small number of known targets (empty train set)
    if(targets_per_tf[i] == 1):
        targets_test = []
        targets_train = targets[0:1]
    else:
        if(len(targets_train)==0):
            targets_test = targets[1:len(targets)]
            targets_train = targets[0:1]
            
    tr.append(targets_train)
    
    print(i, len(targets), len(targets_test), len(targets_train))
    
    y = gex_tf[i,].reshape(-1,1)

    mse = np.zeros((gex_target.shape[0],len(targets_train)))
    rsq = np.zeros((gex_target.shape[0],len(targets_train)))
    mae = np.zeros((gex_target.shape[0],len(targets_train)))
    for j in range(len(targets_train)):
        g2 = targets_train[j]
        x = gex_target[g2,].reshape(-1,1)
        keep = (x > 0) & (y > 0) #remove dropout samples
        if(np.any(keep)):
            model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', max_depth=5, eta=0.1, n_estimators=50, seed=1)
            model.fit(x[keep].reshape(-1,1), y[keep].reshape(-1,1))
    
            for k in range(len(gex_target)):
                x2 = gex_target[k,].reshape(-1,1)
                keep2 = (x2 > 0) & (y > 0) #remove dropout samples
                if(np.any(keep2)):
                    y_pred = model.predict(x2[keep2].reshape(-1,1))
                    mse[k,j] = mean_squared_error(y[keep2].reshape(-1,1), y_pred)
                    mae[k,j] = mean_absolute_error(y[keep2].reshape(-1,1), y_pred)
                    if(np.sum(keep2)>1): #at least two samples needed to compute R2
                        rsq[k,j] = r2_score(y[keep2].reshape(-1,1), y_pred)
        
        #keep best prediction among targets_train        
        mse_pred = np.min(mse, axis=1)
        rsq_pred = np.max(rsq, axis=1)
        mae_pred = np.min(mae, axis=1)
        net_mse[i,:] = mse_pred
        net_rsq[i,:] = rsq_pred
        net_mae[i,:] = mae_pred   

end = time.time()
dt = end - start
print(dt)

#calucalte AUROC
#vectorize results and ground truth   
a_v = adj.reshape((adj.shape[0]*adj.shape[1],1))
n_v = net_mse.reshape((net_mse.shape[0]*net_mse.shape[1],1))
auc1_all = roc_auc_score(a_v, -n_v) #error: smaller is better, use "-" sign
n_v = net_rsq.reshape((net_rsq.shape[0]*net_rsq.shape[1],1))
auc2_all = roc_auc_score(a_v, n_v)
n_v = net_mae.reshape((net_mae.shape[0]*net_mae.shape[1],1))
auc3_all = roc_auc_score(a_v, -n_v) #error: smaller is better, use "-" sign
print(auc1_all)
print(auc2_all)
print(auc3_all)

if(SAVE): 
    pickle.dump((auc1,auc2,auc3,auc1t,auc2t,auc3t,dt,net_mse,net_rsq,net_mae, tr),open('diabetes_' + str(int(100*SV)) + '.pkl','wb'))
