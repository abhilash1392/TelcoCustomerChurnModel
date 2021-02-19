# src/train.py

# Importing the libraries 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import argparse

# Creating the run function
def run(fold):

    # Importing the data 
    df = pd.read_csv('../input/data_folds.csv')

    df = df.sample(frac=1).reset_index(drop=True)

    # Encoding the columns 
    columns = [c for c in df.columns if c not in ('customerID','PhoneService','gender','Dependents',       'Partner','StreamingMovies','StreamingTV','Churn')]
    categoricalColumns = [c for c in columns if df[c].dtype =='object']
    numericalColumns = [c for c in columns if df[c].dtype !='object']
    for c in columns:
        if c not in numericalColumns:
            df[c] = df[c].astype('str')
            lbl = LabelEncoder()
            df[c] = lbl.fit_transform(df[c])

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[columns]
    y_train = df_train.Churn.values
    x_valid = df_valid[columns]
    y_valid = df_valid.Churn.values

    rf= RandomForestClassifier(max_depth=7,n_estimators=1100)
    rf.fit(x_train,y_train)

    y_pred = rf.predict_proba(x_valid)[:,1]

    auc_score = roc_auc_score(y_valid,y_pred)

    print("Fold --> {} | AUC Score --> {:.3f}".format(fold,auc_score))

    joblib.dump(rf,f'../models/dt_{fold}.bin')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',type=int)
    args = parser.parse_args()
    run (fold = args.fold)