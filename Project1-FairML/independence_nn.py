import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
from pathlib import Path

#As a starting point for this class, this example was used: https://github.com/wbawakate/fairtorch/blob/master/examples/example_bar_pass_prediction.py
class DatasetGenerator:
    def __clean_up_data(self, df):
        use_columns = [
            "account_amount",
            "duration",
            "credit_history",
            "credit_purpose",
            "credit_amount",
            "savings_amount",
            "employment_length",
            "installment_rate",
            "other_debtors",
            "residence_length",
            "property",
            "age",
            "installment_plans",
            "housing",
            "existing_credits",
            "job",
            "dependents",
            "telephone",
            "foreign_worker",
            #"credit_score",
            "personal_status",
            "sex"
        ]

        #Making sure the dependent variable has the correct datatype
        #df.loc[:, "personal_status"] = df.loc[:, "personal_status"].astype(str)
        #df.loc[:, "personal_status"] = df.loc[:, "personal_status"].where(df.loc[:, "personal_status"] == "single", 0)
        #df.loc[:, "personal_status"] = df.loc[:, "personal_status"].where(df.loc[:, "personal_status"] != "single", 1)
        #df.loc[:, "personal_status"] = df.loc[:, "personal_status"].astype(int)

        categorical_cols = ["account_amount",
                            "credit_history",
                            "credit_purpose",
                            "savings_amount",
                            "employment_length",
                            "installment_rate",
                            "other_debtors",
                            "residence_length",
                            "property",
                            "installment_plans",
                            "housing",
                            "existing_credits",
                            "job",
                            "dependents",
                            "telephone",
                            "foreign_worker",
                            "personal_status",
                            "sex"]
        df = df.dropna()
        for col in use_columns:
            if col not in categorical_cols:
                df.loc[:, col] = df.loc[:, col].astype(float)

        df.loc[:, "sex"] = df.loc[:, "sex"].astype(str)

        df = df[use_columns]
        df.loc[:, categorical_cols] = df.loc[:, categorical_cols].apply(
            LabelEncoder().fit_transform
        )

        return df.reset_index(drop=True)

    #Handles splitting of the data.
    def train_test_split(self, target_label="credit_score", dataset_csv_path=Path("data/german_credit.csv")):
        df = pd.read_csv(dataset_csv_path)


        y = df[target_label]
        X = df.drop(columns=[target_label])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)

        X_train = self.__clean_up_data(X_train)
        X_test = self.__clean_up_data((X_test))

        sensitive_train = X_train["personal_status"].astype(int)

        return X_train, y_train, X_test, y_test, sensitive_train


class TorchDataset(Dataset):

    def __init__(self, x_df,y_df):
        x = x_df.values
        y = y_df.values
        a = x_df["personal_status"].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.a[idx]


class network(nn.Module):
    def __init__(self, num_train_var):
        super(network,self).__init__()
        self.linear1 = nn.Linear(num_train_var, 128)
        self.batch1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.batch2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64,32)
        self.batch3 = nn.BatchNorm1d(32)
        #self.linear3 = nn.Linear(20,num_y_classes) could be changed if we were doing non-binary classification
        self.linear4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.batch1(x)
        x = F.relu(self.linear2(x))
        x = self.batch2(x)
        x = F.relu(self.linear3(x))
        x = self.batch3(x)
        x = self.linear4(x)
        return x


def validation(loader, model):
    predictions = []
    classifications = []
    actual = []

    model.eval()

    with torch.no_grad():
        for x, y, a in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            actual.append(y.detach().cpu().numpy())

            logit = model(x)

            prediction = torch.sigmoid(logit)
            predictions.append((prediction.detach().cpu()).numpy())

    #flattening lists
    predictions = [item[0] for sublist in predictions for item in sublist]
    actual = [item for sublist in actual for item in sublist]
    classifications = [round(num) for num in predictions]
    return classifications, predictions, actual

if __name__ == "__main__":
    #Load, clean, and split data into pandas
    data_generator = DatasetGenerator()
    X_train, y_train, X_test, y_test, sensitive_train = data_generator.train_test_split()

    #Hyperparameters
    num_train_var = X_train.shape[1]
    num_y_classes = len(set(y_train))
    learning_rate = 0.001
    num_epochs = 500
    use_fairness = True
    alpha = 1000 #scalar for fairness penalty

    #device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    #data to torch
    train_data = TorchDataset(X_train,y_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)

    test_data = TorchDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    #Initialize network
    model = network(num_train_var)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        # Training
        print(f"epoch number {epoch}")
        for batch_idx, (data, targets, sensitive) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device=device)
            targets = targets.to(device=device)

            logit = model(data)

            loss = criterion(logit.view(-1),targets)

            if use_fairness == True:
                prediction = torch.sigmoid(logit)
                preds = prediction.detach().cpu().numpy()
                thresholding = np.array([np.round(num)[0] for num in preds])
                sensitive = sensitive.detach().cpu().numpy().astype(int)

                combined_mean = thresholding.mean()
                single_mean = thresholding[sensitive==1].mean()

                penalty = alpha*(combined_mean-single_mean)**2

                loss = loss + penalty










            loss.backward()

            optimizer.step()

    print("training done")


    classifications, predictions, actual = validation(test_loader, model)

    acc = sum(classifications[i] == actual[i] for i in range(len(classifications)))/len(classifications)
    print(f"accuracy on test set was {acc*100} \%")
    #Adding outputs to the pandas dataframe for test set

    #not sure why X_test["classifications"] = classifications does not work, but it does not, so:
    col1 = pd.Series(predictions)
    col2 = pd.Series(classifications)
    col3 = pd.Series(actual)
    X_test.insert(loc=0, column='predictions', value=col1)
    X_test.insert(loc=0,column='classifications',value=col2)
    X_test.insert(loc=0,column='credit_score',value=col3)

    X_test.index.name = "person_id"
    X_test.to_csv("data/fair_NN_independence_out.csv")

    subset = pd.DataFrame()
    subset.insert(loc=0, column='predictions', value=col1)
    subset.insert(loc=0, column='classifications', value=col2)
    subset.insert(loc=0, column='credit_score', value=col3)
    subset["personal_status"] = X_test["personal_status"]
    subset.index.name = "person_id"
    subset.to_csv("data/subset_fair.csv")


    1+1
    #for epoch in range(num_epochs):

