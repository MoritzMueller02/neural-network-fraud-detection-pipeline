from pipeline import FraudData, FraudDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class FraudNet(nn.Module):
    def __init__(self, n_num, cat_cardinalities, emb_dim=4):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim)
            for num_categories in cat_cardinalities
        ])
        input_dim = n_num + emb_dim * len(cat_cardinalities)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
    
    def forward(self, X_num, X_cat):
        embs = [emb_layer(X_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        x = torch.cat([X_num] + embs, dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))
        return x.squeeze()
    
    def train_model(self, train_loader, epochs=5):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.train()
        for epoch in range(epochs):
            for x_num, x_cat, y in train_loader:
                x_num = x_num.float()
                x_cat = x_cat.long()
                y = y.float()
                optimizer.zero_grad()
                preds = self.forward(x_num, x_cat)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


    def evaluate_model(self, val_loader):
        self.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x_num, x_cat, y = batch
                    y = y.float()
                    all_labels.extend(y.tolist())
                else:
                    x_num, x_cat = batch
                    y = None
                x_num = x_num.float()
                x_cat = x_cat.long()
                probs = self.forward(x_num, x_cat)
                preds = (probs > 0.5).int()
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
        if all_labels:
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds[:len(all_labels)])
            all_probs = np.array(all_probs[:len(all_labels)])
            acc  = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec  = recall_score(all_labels, all_preds, zero_division=0)
            f1   = f1_score(all_labels, all_preds, zero_division=0)
            auc  = roc_auc_score(all_labels, all_probs)
            print(f"Validation Metrics:")
            print(f"  Accuracy  : {acc:.4f}")
            print(f"  Precision : {prec:.4f}")
            print(f"  Recall    : {rec:.4f}")
            print(f"  F1 Score  : {f1:.4f}")
            print(f"  ROC-AUC   : {auc:.4f}")
        else:
            print("No labels in dataset, skipping metrics.")



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
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    cat_cardinalities = [train["ProductCD"].nunique(), 
                            train["card4"].nunique()]
    
    model = FraudNet(n_num=len(num_cols), 
                     cat_cardinalities=cat_cardinalities, emb_dim=4)
    
    model.train_model(train_loader, epochs=1)
    
    model.evaluate_model(test_loader)
