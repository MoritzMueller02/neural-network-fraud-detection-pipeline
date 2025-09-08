import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import kaggle
import shutil
import os


class FraudData:
    def __init__(self, train_path, test_path, nrows=None):
        
        if os.path.exists('data') and os.path.isdir('data'):
            self.train = pd.read_csv(train_path, nrows=nrows)
            self.test = pd.read_csv(test_path, nrows=nrows)
        else:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('ieee-fraud-detection', path='data', unzip=True)
            extract_dir = "D:/Repos/NEURAL NETWORK SIM/data/"
            shutil.unpack_archive("ieee-fraud-detection.zip", extract_dir)

    @staticmethod
    def reduce_cols(train, test, cols_to_keep: list):
        test_cols = [col for col in cols_to_keep if col in test.columns]
        return train.loc[:, cols_to_keep], test.loc[:, test_cols]


    @staticmethod
    def label_encode(train, test, cat_cols: list):
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
        return train, test

    @staticmethod
    def scale_numeric(train, test, num_cols: list):
        scaler = StandardScaler()
        scaler.fit(pd.concat([train[num_cols], test[num_cols]], axis=0))
        train[num_cols] = scaler.transform(train[num_cols])
        test[num_cols] = scaler.transform(test[num_cols])
        return train, test


class FraudDataset(Dataset):
    def __init__(self, df, target_col=None):
        self.X_num = df[["TransactionDT", "TransactionAmt"]].values.astype("float32")
        self.X_cat = df[["ProductCD", "card4"]].values.astype("int64")
        self.y = df[target_col].values.astype("float32") if target_col else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        X_num = torch.tensor(self.X_num[idx])
        X_cat = torch.tensor(self.X_cat[idx])
        if self.y is not None:
            y = torch.tensor(self.y[idx])
            return X_num, X_cat, y
        return X_num, X_cat


if __name__ == "__main__":

    train_path = "data/train_transaction.csv"
    test_path = "data/test_transaction.csv"
    
    data = FraudData(train_path, test_path, nrows=10000)
    
    cols_to_keep = ['isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card4']
    train, test = FraudData.reduce_cols(data.train, data.test, cols_to_keep)
    
    cat_cols = ['ProductCD', 'card4']
    train, test = FraudData.label_encode(train, test, cat_cols)

    num_cols = ['TransactionDT', 'TransactionAmt']
    train, test = FraudData.scale_numeric(train, test, num_cols)

    train_ds = FraudDataset(train, target_col="isFraud")
    test_ds = FraudDataset(test)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    for X_num, X_cat, y in train_loader:
        print("Numerical:", X_num.shape) #tensor size (32, 2) -> rows, feautres
        print("Categorical:", X_cat.shape)
