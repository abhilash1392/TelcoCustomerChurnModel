# src/create_folds.py 

import numpy as np 
import pandas as pd 
from  sklearn.model_selection import StratifiedKFold 


if __name__ == "__main__":
    # Reading the dataframe 
    df = pd.read_csv('../input/data.csv')

    df['kfold'] = -1 

    df = df.sample(frac=1).reset_index(drop=True)

    # We are going to do some data manipulation here
    # In eda we have seen tham some values of feature TotalCharges were empty we are going to fill them with the   median value having same monthly charges and same contract 
    df['TotalCharges'] = df['TotalCharges'].replace(' ',0)
    df.TotalCharges = df.TotalCharges.astype(float)
    for idx, i in enumerate(df.TotalCharges):
        if i == 0:
            df.loc[idx,'TotalCharges'] = np.median(df['TotalCharges'].astype(float)[(df.Contract == df.Contract[idx]) & df. TotalCharges!=0 & (df.MonthlyCharges == df.MonthlyCharges[idx])])

    # Changing the Churn values from 'Yes':1 and 'No':0
    df.Churn = df.Churn.map({'No':0,'Yes':1})
    # Getting the targets 
    y = df.Churn.values 

    # Creating a kfold Stratifield object 
    kf = StratifiedKFold(n_splits=5)

    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f 

    
    # saving the new dataframe 
    df.to_csv('../input/data_folds.csv')

    print('Done Done Done')

     

    
