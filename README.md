# Telco Customer Churn

## Focused customer retention programs

# Description

## Context
"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]

## Content
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The data set includes information about:

- Customers who left within the last month – the column is called Churn

- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup,  device protection, tech support, and streaming TV and movies

- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing monthly charges, and total charges

- Demographic info about customers – gender, age range, and if they have partners and dependents

## Feature Selection
Please look at the file notebooks/eda.ipynb to view exploratory data analysis, data processing and data visualization.

## Directories 

 - input --> Having the data CSV 

 - notebooks --> Having Jupyter Notebook

 - src --> Having Python Scripts

 - models --> Having saved models

## Result

### Using LabelEncoder and RandomForest

Fold --> 0 | AUC Score --> 0.837

Fold --> 1 | AUC Score --> 0.836

Fold --> 2 | AUC Score --> 0.837

Fold --> 3 | AUC Score --> 0.844

Fold --> 4 | AUC Score --> 0.874

### Using OneHotEncoder and RandomForest

Fold --> 0 | AUC Score --> 0.837

Fold --> 1 | AUC Score --> 0.835

Fold --> 2 | AUC Score --> 0.839

Fold --> 3 | AUC Score --> 0.844

Fold --> 4 | AUC Score --> 0.872