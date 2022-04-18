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


import argparse
import crypten

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from Load_data2 import Import_Data
from models.Update import Ratio_Cross_Entropy
from models.Fed import whole_determination, outlier_detect

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1,2)
        x = self.layer_input(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

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


def divide_trainset_to_client(train_set, cli_num,batch_size):
    length_list = []
    train_sets = []
    for _ in range(cli_num - 1):
        length_list.append(len(train_set) // cli_num)
    length_list.append(len(train_set) - (cli_num - 1)* (len(train_set) // cli_num))
    train_sets_pre = random_split(train_set, length_list)
    for i in train_sets_pre:
        train_sets.append(DataLoader(i,batch_size=batch_size))
    return train_sets


def define_network(cli_num,lr_,input_shape):
    createVar = locals()
    optimizers = []
    models = []
    params = []
    for i in range(cli_num):
        k = str(i + 1)
        model_name = 'model_' + k
        opti_name = 'optimizer_' + k
        createVar[model_name] = Net(input_shape).to(device)
        createVar[opti_name] = optim.SGD(
            locals()[model_name].parameters(),
            lr=lr_)
        models.append(locals()[model_name])
        params.append(list(locals()[model_name].parameters()))
        optimizers.append(locals()[opti_name])
    return models, optimizers, params


def fl_train(train_sets, fl_models, fl_optimizers, params):
    new_params = list()
    for k in range(len(train_sets)):
        for data, target in (train_sets[k]):
            fl_optimizers[k].zero_grad()
            data, target = data.to(device), target.to(device)
            output = fl_models[k](data)
            #loss = criterion(output, target.unsqueeze(1))# version 1
            loss = criterion(output, target)
            loss.backward()
            fl_optimizers[k].step()
    for param_i in range(len(params[0])):
        fl_params = list()
        for remote_index in range(CLI_NUM):
            clone_param = params[remote_index][param_i].clone().cpu()
            #fl_params.append(crypten.cryptensor(torch.tensor(clone_param)))
            #fl_params.append(torch.tensor(clone_param))
            #fl_params.append(crypten.cryptensor(clone_param.clone().detach()))
            fl_params.append(clone_param.clone().detach())
        sign = 0
        for i in fl_params:
            if sign == 0:
                fl_param = i
                sign = 1
            else:
                fl_param = fl_param + i

        #new_param = (fl_param / CLI_NUM).get_plain_text()
        new_param = (fl_param / CLI_NUM)
        new_params.append(new_param)

    with torch.no_grad():
        for model_para in params:
            for param in model_para:
                param *= 0

        for remote_index in range(CLI_NUM):
            for param_index in range(len(params[remote_index])):
                new_params[param_index] = new_params[param_index].to(device)
                params[remote_index][param_index].set_(new_params[param_index])
    return fl_models

def train(train_loader, model,optimizer):
    model.train()
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate (train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)
            print("pred",y_pred.shape)
            print("pred",y_batch.shape)

            #loss = criterion(y_pred, y_batch.unsqueeze(1)) #version 1
            #acc = binary_acc(y_pred, y_batch.unsqueeze(1)) # version 1
            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
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
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    CLI_NUM = 20
    
    Data = Import_Data("../../2010_out_new.csv")
    print (Data.df.head())
    X_train, X_test, y_train, y_test = train_test_split(Data.X, Data.y, test_size=0.20)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    #y_train = torch.tensor(y_train, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.int64)#new for 2-dim y
    y_test = torch.tensor(y_test.values, dtype = torch.int64)#new for 2-dim y
    train_set = TensorDataset(X_train, y_train)
    #train_set = TensorDataset(X_train, y_train)
    test_data = TestData(torch.FloatTensor(X_test))

    #cls_weights = torch.from_numpy(compute_class_weight(class_weight="balanced",classes=np.unique(y_train.numpy()),y=y_train.numpy()))

    #weights = cls_weights[y_train.long()]
    #sampler = WeightedRandomSampler(weights, len(y_train), replacement=True)

    train_sets = divide_trainset_to_client(train_set, cli_num=CLI_NUM,batch_size = BATCH_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #model = Net(X_train.shape[1])
    model = MLP(dim_in=X_train.shape[1], dim_hidden=32, dim_out=2)
    model.to(device)
    print(model)
    #criterion = nn.BCEWithLogitsLoss()
    criterion=Ratio_Cross_Entropy(device,class_num=2, alpha=None, size_average = False)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    #train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    models, optimizers, _ = define_network(cli_num=CLI_NUM, lr_=LEARNING_RATE,input_shape = X_train.shape[1])
    fl_models, fl_optimizers, params = define_network(
    cli_num=CLI_NUM, lr_=LEARNING_RATE,input_shape = X_train.shape[1])

with open('./result/result.txt', 'w') as f:
    for n in range(CLI_NUM):
        model = train(train_sets[n], models[n], optimizers[n])
        test(test_loader, y_test, model)
        print("centralized classification report")
        test(test_loader, y_test, model)

    for i in range(EPOCHS):
        print("FEDERATED TRAINING")
        fl_models = fl_train(train_sets, fl_models, fl_optimizers, params)
    #test(test_loader, y_test, fl_models[0])
        print("federated classification_report")
        if (i %20==0):
            test(test_loader, y_test, fl_models[0])
            """fl_models[0].eval()
            y_pred_list = []
            with torch.no_grad():
                for X_batch in test_loader:
                    X_batch = X_batch.to(device)
                    y_test_pred = model(X_batch)
                    y_test_pred = torch.sigmoid(y_test_pred)
                    y_pred_tag = torch.round(y_test_pred)
                    y_pred_list.append(y_pred_tag.cpu().numpy())

            y_pred_list = [a.squeeze().tolist() for a in y_pred_list]"""
    print("outer loop")
    test(test_loader, y_test, fl_models[0])
            
            
