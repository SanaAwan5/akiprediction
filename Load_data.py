import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold

class Import_Data:
    def __init__(self,file_name):
        self.df = pd.read_csv(file_name)
        self.df.drop(['ID'],axis=1,inplace=True)
        #self.y = self.df['Label1']
        self.df['label'] = self.df['label'].astype(str).map({'0':0,'1':1,'2':1,'3':1})
        self.df['label'] = self.df['label'].astype(int)
        self.y = self.df.pop('label')
        self.X = self.df   ##DataFrames are mutable. Inplace assignment
        self.kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    """def cv_train():
        # The folds are made by preserving the percentage of samples for each class.
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cnt = 1
        # split()  method generate indices to split data into training and test set.
        for train_index, test_index in kf.split(self.X, self.y):
            print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
            cnt+=1"""
