import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from models.Update import Ratio_Cross_Entropy, FocaLoss
from models.Fed import whole_determination, outlier_detect
import time



import argparse
import crypten

import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from Load_data import Import_Data
import decimal
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma,device):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = FocaLoss(device,logits,labels_one_hot, weights, gamma,True)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=2, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


def build_classes_dict(trainset):
    classes = {}
    #y_train = y_train.numpy()
    for ind,d in enumerate(trainset):
        _,label = d
        label = int(label)
        #print(label)
        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]
    #print(classes.keys())
    return classes

def sample_dirichlet_train_data(classes,no_participants, alpha=1000):
    """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
    """

 
    _,class_sample_count = np.unique(y_train,return_counts=True)
    print(class_sample_count)
        
    class_size = len(classes[0]) #for cifar: 5000
    print(class_size)
    per_participant_list = defaultdict(list)
    no_classes =2 # for cifar: 10

    sample_nums = []
    for n in range(no_classes):
        sample_num = []
        random.shuffle(classes[n])
        sampled_probabilities =(len(classes[n]))*np.random.dirichlet(np.array(no_participants*[alpha]))
        for user in range(no_participants):
            no_samples = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][:min(len(classes[n]), no_samples)]
            sample_num.append(len(sampled_list))
            per_participant_list[user].extend(sampled_list)
            classes[n] = classes[n][min(len(classes[n]), no_samples):]
        sample_nums.append(sample_num)
        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
    return per_participant_list

def get_train(train_dataset,indices):
    #train_sets = []
    train_loaders=DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=SubsetRandomSampler(
                                               indices),pin_memory=True, num_workers=1)

    #train_loaders = DataLoader(train_dataset,batch_sampler= StratifiedBatchSampler(y_train[indices],batch_size=BATCH_SIZE))

    #print(len(train_sets))
    return train_loaders

"""class BinaryClassification(nn.Module):
    def __init__(self,input_shape):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(input_shape, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x"""

class BinaryClassification(nn.Module):
    def __init__(self,input_shape):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(input_shape, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128,64)
        self.layer_4 = nn.Linear(64,64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.5)
       
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x= self.relu(self.layer_3(x))
        x= self.batchnorm3(x)
        x = self.dropout3(x)
        x= self.relu(self.layer_4(x))
        x= self.batchnorm4(x)
        x = self.dropout4(x)
        x = self.layer_out(x)

        return x



class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(input_shape, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

def float_range(start,stop,step):
    temp = []
    while start < stop:
        temp.append(float(start))
        start = decimal.Decimal(start) + decimal.Decimal(step)
    return temp
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def AUC_Score(model,testloader,y_test):
    auc_plot = []
    preds=[]
    th_range = float_range(-0.1,1.2,'0.1')
        #prob = model.predict_proba(X_test)
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            #y_pred_tag = torch.round(y_test_pred)
            preds.append(y_test_pred.cpu().numpy())
    preds = [a.squeeze().tolist() for a in preds]
    for th in th_range:
        y_pred = [0 if e < th else 1 for e in list(preds)]
        tn,fp,fn,tp = metrics.confusion_matrix(y_test,y_pred).ravel()
        tpr = tp / (tp+fn)
        fpr = fp / (fp+tn)
        auc_plot.append((tpr,fpr))
    tpr_temp = [a_tuple[0] for a_tuple in auc_plot]
    fpr_temp = [a_tuple[1] for a_tuple in auc_plot]
    auc_score = "{:.2f}".format(metrics.auc(fpr_temp,tpr_temp))
    return auc_score


class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)
## test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__ (self):
        return len(self.X_data)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def divide_trainset_to_client(train_set,cli_num,batch_size):
    length_list = []
    train_sets = []
    for _ in range(cli_num - 1):
        length_list.append(len(train_set) // cli_num)
    length_list.append(len(train_set) - (cli_num - 1)* (len(train_set) // cli_num))
    train_sets_pre = random_split(train_set, length_list)
    
    for i in train_sets_pre:
        train_sets.append(DataLoader(i,batch_size=BATCH_SIZE))
    return train_sets


def define_network(cli_num,lr_,weight_decay,input_shape):
    createVar = locals()
    optimizers = []
    models = []
    params = []
    for i in range(cli_num):
        k = str(i + 1)
        model_name = 'model_' + k
        opti_name = 'optimizer_' + k
        createVar[model_name] = BinaryClassification(input_shape).to(device)
        createVar[opti_name] = optim.SGD(
            locals()[model_name].parameters(),
            lr=lr_,weight_decay=weight_decay)
        models.append(locals()[model_name])
        params.append(list(locals()[model_name].parameters()))
        optimizers.append(locals()[opti_name])
    return models, optimizers, params


def fl_train(train_sets, fl_models, fl_optimizers, params):
    new_params = list()
    count = np.zeros(len(train_sets))
    for k in range(len(train_sets)):
        #print('len',len(train_sets))
        for batch_idx,(data,target) in enumerate(train_sets[k]):
            count[k] =count[k] + torch.count_nonzero(target,dim=-1)
            fl_optimizers[k].zero_grad()
            data, target = data.to(device), target.to(device)
            output = fl_models[k](data)
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()
            fl_optimizers[k].step()
    print("Number of positive samples for clients", count)
    for param_i in range(len(params[0])):
        fl_params = list()
        for remote_index in range(CLI_NUM):
            clone_param = params[remote_index][param_i].clone().cpu()
            #fl_params.append(crypten.cryptensor(clone_param.clone().detach().requires_grad_(True)))
            #print("time to encrypt", end - start)
            #fl_params.append(torch.tensor(clone_param))
            #fl_params.append(crypten.cryptensor(clone_param.clone().detach()))
            #fl_params.append(clone_param.clone().detach().requires_grad_(True))
            fl_params.append(clone_param.clone().detach().requires_grad_(True))
        sign = 0
        start_aggregation = time.time()
        for i in fl_params:
            if sign == 0:
                fl_param = i
                sign = 1
            else:
                fl_param = fl_param + i

        #new_param = (fl_param / CLI_NUM).get_plain_text()
        new_param = (fl_param / CLI_NUM)
        new_params.append(new_param)
        end_aggregation = time.time()
        #print("total time for aggregation", end_aggregation - start_aggregation)
    with torch.no_grad():
        for model_para in params:
            for param in model_para:
                param *= 0

        for remote_index in range(CLI_NUM):
            for param_index in range(len(params[remote_index])):
                new_params[param_index] = new_params[param_index].to(device)
                params[remote_index][param_index].set_(new_params[param_index])
    return fl_models

def train(epoch,dataloader, model,optimizer):
    model.train()
    #for e in range(1, epochs+1):
    epoch_loss = 0
    epoch_acc = 0
    for batch_idx, (X_batch, y_batch) in enumerate (train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
            
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        #loss=criterion(y_pred,y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
    return model

def test(test_loader, y_test, model):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print(classification_report(y_test, y_pred_list,zero_division=0))
    


    
if __name__ == "__main__":
    EPOCHS = 300
    BATCH_SIZE = 120
    LEARNING_RATE = 0.001
    CLI_NUM = 8
    WEIGHT_DECAY=0.001
    DIRICHLET_ALPHA = 1
    print('DIRICHLET ALPHA', DIRICHLET_ALPHA)
    
    Data = Import_Data('../../2010_out_new.csv')

    print (Data.df.head())
    train_sets=[]
    #crypten.init()
    
    X_train, X_test, y_train, y_test = train_test_split(Data.X, Data.y, train_size=13000,random_state = 42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train,dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_data = TrainData(X_train,y_train)
    #train_data_auc=TestData(torch.FloatTensor(X_train))
    test_data = TestData(torch.FloatTensor(X_test))

    # change from torch.float32
    #train_set = TrainData(X_train, y_train)
    #train_set = TensorDataset(X_train, y_train)
    print ("Before",X_train.shape)
    print (y_train.shape)


    #######Weighted Random Sampling ######################## 
    #class_sample_count = torch.tensor([(y_train==t).sum() for t in torch.unique(y_train, sorted=True)])
    #_,class_sample_count = np.unique(y_train, return_counts=True)
    #class_weights = 1 / torch.tensor(class_sample_count, dtype=torch.float)
    #sample_weights = torch.tensor([class_weights[t] for t in y_train.long()])
    #train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    #######sampler is mutually exclusive with shuffle in train_loader#############################
    #train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    #train_loader = DataLoader(dataset=train_data,batch_sampler= StratifiedBatchSampler(y_train,batch_size=BATCH_SIZE))
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    #train_loader_auc= DataLoader(dataset=train_data_auc, batch_size=BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('count', class_sample_count)
    #print('weights',sample_weights)
    #models, optimizers, _ = define_network(cli_num=CLI_NUM, lr_=LEARNING_RATE,weight_decay=WEIGHT_DECAY,input_shape = X_train.shape[1])
    fl_models, fl_optimizers, params = define_network(
    cli_num=CLI_NUM, lr_=LEARNING_RATE,weight_decay=WEIGHT_DECAY,input_shape = X_train.shape[1])

    #model = BinaryClassification(X_train.shape[1])
    #model.to(device)
    #print(model)
    #criterion=FocaLoss(device,BATCH_SIZE, class_num=2, alpha=None, gamma=2, size_average=True)
    weight = torch.tensor([4.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    #criterion=Ratio_Cross_Entropy(device,class_num=2, alpha=None, size_average = True)
    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    #scheduler: decays the learning rate of each parameter group by gamma every step_size epochs
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)Data = Import_Data("../../2010_out_new.csv")
    classes_dict = build_classes_dict(train_data)
    print('build_classes_dict done')
   
    ## sample indices for participants using Dirichlet distribution
    indices_per_participant = sample_dirichlet_train_data(classes_dict,CLI_NUM,DIRICHLET_ALPHA)
    for pos,indices in indices_per_participant.items():
        #print(pos)
        #print(indices)
        train_sets.append(get_train(train_data,indices))
    #train_sets=divide_trainset_to_client(train_set,CLI_NUM,BATCH_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('train loaders done')
    print('length train_sets',len(train_sets))
    print(device)
    #for n in range(CLI_NUM):
    print('criterion', criterion)
    """X_train_new = X_train.reshape(CLI_NUM,-1,X_train.shape[1])
    y_train_new = y_train.reshape(CLI_NUM,-1)

    print ("After",X_train_new.shape)
    print (y_train_new.shape)

    train_loader_AC = []

    for i in range(CLI_NUM):
        train_data_new= TrainData(X_train_new[i],y_train_new[i])
        train_loader_AC.append( DataLoader(dataset=train_data_new, batch_size=BATCH_SIZE, shuffle=True))
    for i in range (EPOCHS):
        model_new=train(i,train_loader,model,optimizer)
        #scheduler.step()
        if (i% 10 ==0):
            test(test_loader,y_test, model_new)
            #print ("Auc Score 2010_train: ",AUC_Score(model,train_loader_auc,y_train))
            print ("Auc Score 2010_validation: ",AUC_Score(model,test_loader,y_test))
        
    print("outer loop")
    print ("Auc Score 2010_validation_centralized_setting:",AUC_Score(model,test_loader,y_test))"""

    for i in range(EPOCHS):
        #print("FEDERATED TRAINING")
        fl_models = fl_train(train_sets, fl_models, fl_optimizers, params)
        if (i%20==0):
            test(test_loader, y_test, fl_models[0])
    #test(test_loader, y_test, fl_models[0])
        #print("federated classification_report")
        #if (i %20==0):
        #test(test_loader, y_test, fl_models[0])         
    print("outer loop")
    #test(test_loader, y_test, fl_models[0])
    print ("Auc Score 2010_validation_federated_setting:",AUC_Score(fl_models[0],test_loader,y_test))