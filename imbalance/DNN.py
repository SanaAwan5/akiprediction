
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from Load_data import Import_Data
import decimal
## train data

def float_range(start,stop,step):
    temp = []
    while start < stop:
        temp.append(float(start))
        start = decimal.Decimal(start) + decimal.Decimal(step)
    return temp

class BinaryClassification(nn.Module):
    def __init__(self,input_shape):
        super(BinaryClassification, self).__init__()
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

if __name__ == "__main__":
    Data = Import_Data("../../2010_out_new.csv")

    print (Data.df.head())
    #print ("Nan Columns: ",Data.df.columns[Data.df.isna().any()].tolist())
    #print (Data.df.columns[100:150])

    X_train, X_test, y_train, y_test = train_test_split(Data.X, Data.y, test_size=0.20)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_data = TrainData(torch.FloatTensor(X_train),torch.FloatTensor(y_train))
    test_data = TestData(torch.FloatTensor(X_test))

    EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model = BinaryClassification(X_train.shape[1])
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()


        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
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
    print(classification_report(y_test, y_pred_list))
    print ("Auc Score 2010_validation: ",AUC_Score(model,test_loader,y_test))
