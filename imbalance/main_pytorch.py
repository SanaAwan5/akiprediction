#from Load_data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset,DataLoader,random_split
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
from sklearn.utils.class_weight import compute_class_weight

#from sklearn import ensemble
#from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.model_selection import train_test_split
#from collections import Counter
#from xgboost import XGBClassifier
#import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
#import torch.optim.lr_scheduler.StepLR
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from Load_data import Import_Data
import decimal

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

def float_range(start,stop,step):
    temp = []
    while start < stop:
        temp.append(float(start))
        start = decimal.Decimal(start) + decimal.Decimal(step)
    return temp

def AUC_Score(model,test_loader,y_test):
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

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def normalize(target_array, data_max, data_min):
    return (target_array - data_min) / (data_max - data_min)




def logarithmic(target_array):
    return np.log(target_array + 1)


def calculate_r_square(output, target):
    a = torch.sum((output - target).pow(2))
    b = torch.sum((target - target.mean()).pow(2))
    return 1 -torch.div(a,b,rounding_mode = None)

def preprocess(X):
    #train_data = pd.read_csv(train_file)
    #test_data = pd.read_csv(test_file)
    y = X.pop('label')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #train_x = x_train_origin
    #train_y = y_train_origin
    #test_x = x_test_origin
    #test_y = y_test_origin
    #train_x = torch.tensor(X_train, dtype=torch.float32)
    #train_y = torch.tensor(y_train.to_numpy(dtype=np.float32))
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train.values, dtype = torch.int64)

    #train_y = train_y.long()
    
    test_x = torch.tensor(X_test, dtype=torch.float32)
    #test_y = torch.tensor(y_test.to_numpy(dtype=np.float32))
    test_y = torch.tensor(y_test.values, dtype= torch.int64)

    """cls_weights = torch.from_numpy(compute_class_weight(class_weight="balanced",classes = np.unique(np.ravel(train_y.numpy())),y= np.ravel(train_y.numpy())))
    
    weights = cls_weights[train_y.long()]
    sampler = WeightedRandomSampler(weights, len(train_y), replacement=True)"""
    trainset = TensorDataset(train_x, train_y)
    return trainset,train_x,train_y, test_x, test_y



class Setting:
    """Parameters for training"""

    def __init__(self):
        self.epoch = 5000
        self.lr = 0.05
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.batch_size = 200

class Net(nn.Module):
    def __init__(self,input_size):
        super(Net, self).__init__()        # Number of input features is 12.
        self.layer_1 = nn.Linear(input_size, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x




"""class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,50)
        #self.fc4 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(50, 1)
        #self.dropout1 = nn.Dropout(p=0.25)
        #self.dropout2 = nn.Dropout(p=0.25)
        #self.dropout3 = nn.Dropout(p=0.25)
        #self.dropout4 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.log_softmax(self.fc4(x), dim=1)
        x = self.fc4(x)
        x = F.log_softmax(x, dim =1)
        return x"""

def train(epoch,dataloader, model,optimizer):
    """model.train()
    #print("here")
    for batch_idx, (data, target) in enumerate(dataloader):
        #print("here", flush=True)
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        #weight = torch.tensor([.1,10])
        loss = criterion(output, target.unsqueeze(1))
        #loss = loss * weight
        #loss=loss.mean()
        loss.backward()
        optimizer.step()
    return model"""
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



def test(test_x, test_y, model):
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
    """df = pd.read_csv('../AKI_data/2010/2010_out_new.csv')
    df.drop(['ID'],axis=1,inplace=True)

    df['label'] = df['label'].astype(str).map({'0':0,'1':1,'2':1,'3':1})
    df['label'] = df['label'].astype(int)
    #print(df['label'].dtype)
    #df['split'] = np.random.randn(df.shape[0],1)
    print(df.groupby('label').count())
    #msk = np.random.rand(len(df)) <=0.8"""
    Data = Import_Data('../AKI_data/2010/2010_out_new.csv')

    print (Data.df.head())
    #print ("Nan Columns: ",Data.df.columns[Data.df.isna().any()].tolist())
    #print (Data.df.columns[100:150])
    epochs = 200
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.001
    batch_size = 200

    X_train, X_test, y_train, y_test = train_test_split(Data.X, Data.y, test_size=0.20)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_data = TrainData(torch.FloatTensor(X_train),torch.FloatTensor(y_train))
    train_data_auc=TestData(torch.FloatTensor(X_train))
    test_data = TestData(torch.FloatTensor(X_test))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    train_loader_auc= DataLoader(dataset=train_data_auc, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model = BinaryClassification(X_train.shape[1])
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    #scheduler: decays the learning rate of each parameter group by gamma every step_size epochs
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    

    #train_data = df[msk]
    #test_data = df[~msk]
    

    #train_data.to_csv('train_data.csv',index = False)
    #test_data.to_csv('test_data.csv',index=False)
    #trainset,train_x,train_y, test_x, test_y = preprocess(df)
    

    #trainloader = DataLoader(trainset,batch_size=batch_size)
    #train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    #testloader = DataLoader(dataset=test_data, batch_size=1)


    #trainloader = DataLoader(trainset,batch_size=128)
    #print ("Nan Columns: ",Data.df.columns[Data.df.isna().any()].tolist())
    #print (Data.df.columns[100:150])
    #hyper parameters
    weight = torch.tensor([1,4])

    #posi_weight=torch.Tensor([(12452/2629)])
    #criterion = nn.BCEWithLogitsLoss(pos_weight=posi_weight)
    #criterion=nn.BCEWithLogitsLoss()
    #model =Net(input_size = test_x.shape[1])
    #optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    #scaler = sc.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    
    for i in range (epochs):
        model_new=train(i,train_loader,model,optimizer)
        #scheduler.step()
        if (i% 10 ==0):
            test(test_loader,y_test, model_new)
            print ("Auc Score 2010_train: ",AUC_Score(model,train_loader_auc,y_train))
            print ("Auc Score 2010_validation: ",AUC_Score(model,test_loader,y_test))
        
    print("outer loop")
    print ("Auc Score 2010_validation: ",AUC_Score(model,test_loader,y_test))
