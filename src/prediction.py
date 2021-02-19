# src/prediction.py

# Importing the libraries 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import argparse
import pickle
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Creating the run function

# print('Enter the values of the model input ')
# print({'SeniorCitizen': '0 for young 1 for old',
#  'tenure': 'EnterNumericalValue',
#  'MultipleLines': ['No phone service', 'No', 'Yes'],
#  'InternetService': ['DSL', 'No', 'Fiber optic'],
#  'OnlineSecurity': ['No internet service', 'No', 'Yes'],
#  'OnlineBackup': ['No', 'Yes', 'No internet service'],
#  'DeviceProtection': ['No internet service', 'No', 'Yes'],
#  'TechSupport': ['No internet service', 'No', 'Yes'],
#  'Contract': ['Two year', 'Month-to-month', 'One year'],
#  'PaperlessBilling': ['No', 'Yes'],
#  'PaymentMethod': ['Credit card (automatic)',
#   'Electronic check',
#   'Mailed check',
#   'Bank transfer (automatic)'],
#  'MonthlyCharges': 'EnterNumericalValue',
#  'TotalCharges': 'EnterNumericalValue'})




def model_prediction(input_value):


    pickle_in = open('../models/of_4.pkl','rb')
    model = pickle.load(pickle_in)

    input_value = list(input_value)


    columns = ['SeniorCitizen',
            'tenure',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod',
            'MonthlyCharges',
            'TotalCharges']
    model_input = {}

    for i in range(len(columns)):
        model_input[columns[i]] = input_value[i]

    model_df = pd.DataFrame(model_input,index=[0])

    columns =  [c for c in model_df.columns]
    categoricalColumns = [c for c in columns if model_df[c].dtype =='object']
    numericalColumns = [c for c in columns if model_df[c].dtype !='object']

    y_pred = model.predict(model_df[columns])
    

    return y_pred


model_prediction([0, 72, 'Yes', 'No', 'No internet service', 'No internet service',
        'No internet service', 'No internet service', 'Two year', 'No',
        'Credit card (automatic)', 25.45, 1866.45])