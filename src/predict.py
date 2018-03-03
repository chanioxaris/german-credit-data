from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv


# Open csv dataset files and read data
df_train = pd.read_csv('dataset_train.tsv', sep='\t')
df_pred = pd.read_csv('dataset_predict.tsv', sep='\t')

# If qualitative data encode them using LabelEncoder
# If numerical and continuous split them to maximum five buckets
encoded_data_train = list()
encoded_data_pred = list()

for attr in df_train.columns[:-2]:
    if (type(df_train[attr][0]) is str):    
        values = (df_train[attr]).value_counts().sort_index().index
        encoder = LabelEncoder().fit(values)
        
        encoded_col_train = encoder.transform(df_train[attr]).tolist()       
        encoded_col_pred = encoder.transform(df_pred[attr]).tolist()        
    else:  
        if len(list(set(df_train[attr]))) > 5:      
            encoded_col_train = pd.cut(df_train[attr], bins = 5, labels=[0,1,2,3,4]).tolist()            
            encoded_col_pred = pd.cut(df_pred[attr], bins = 5, labels=[0,1,2,3,4]).tolist() 
        else:
            encoded_col_train = df_train[attr]
            encoded_col_pred = df_pred[attr]
                        
    encoded_data_train.append(encoded_col_train)
    encoded_data_pred.append(encoded_col_pred)

# Filter data and keep only the attributes we are interested in
data_train = list()
data_pred = list()

attributes = [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 14, 19]

for attr in attributes:        
    data_train.append(encoded_data_train[attr]) 
    data_pred.append(encoded_data_pred[attr]) 


data_train = np.transpose(np.array(data_train))
data_pred = np.transpose(np.array(data_pred))

# Use RandomForestClassifier to predict the dataset
classifier_RF = RandomForestClassifier().fit(data_train, df_train.Label)
    
yPred_RF = classifier_RF.predict(data_pred)

# Export predictions to csv file
with open('Predictions.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, delimiter= '\t', quoting=csv.QUOTE_NONE)
    wr.writerow(["Client_ID", "Predicted_Label"])
    for pred_id, predict in zip(df_pred.Id, yPred_RF):
        if predict == 1:           
            csv_out = [str(pred_id), "Good"]
        else:
            csv_out = [str(pred_id), "Bad"]
        wr.writerow(csv_out)  
#========================================================================================================#