from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import csv


# Calculate entropy of whole dataset
def entropy_dataset(data):
    good, bad = data.value_counts()
      
    freq_good = good/(good + bad)
    freq_bad = bad/(good + bad)

    return  -(freq_good * math.log(freq_good, 2) + freq_bad * math.log(freq_bad, 2))


# Calculate entropy of each attribute
def entropy_subset(data):
    subset_entr = 0
    
    good = dict.fromkeys(list(set(data)), 0)
    bad = dict.fromkeys(list(set(data)), 0)
    
    for i in range(len(data)):
        if df.Label[i] == 1:
            good[data[i]] += 1
        else:
            bad[data[i]] += 1
    
    for i in list(set(data)):
        freq_good = good[i]/(good[i] + bad[i])
        freq_bad = bad[i]/(good[i] + bad[i])
        
        weight = ((good[i] + bad[i])/len(data))
        
        subset_entr += weight * -(freq_good * math.log(freq_good, 2) + freq_bad * math.log(freq_bad, 2))       
        
    
    return subset_entr



df = pd.read_csv('train.tsv', sep='\t')

# If qualitative data encode them using LabelEncoder
# If numerical and continuous split them to maximum five buckets
encoded_data = list()

for attr in df.columns[:-2]:
    if (type(df[attr][0]) is str):    
        values = (df[attr]).value_counts().sort_index().index

        encoder = LabelEncoder().fit(values)
        encoded_col = encoder.transform(df[attr]).tolist()        
    else:  
        if len(list(set(df[attr]))) > 5:      
            encoded_col = pd.cut(df[attr], bins = 5, labels=[0,1,2,3,4]).tolist()      
        else:
            encoded_col = df[attr]
            
    encoded_data.append(encoded_col)    


# Calculate information gain for each attribute
entr = entropy_dataset(df.Label)

info = dict()

for attr in range(len(encoded_data)):  

    info[attr] = entr - entropy_subset(encoded_data[attr])

# Sort attributes by information gain
sorted_info = sorted(info.items(), key=lambda x: x[1])
    
attribute = list(range(len(encoded_data)))

info_gain_acc = list()


# Remove one attribute per loop and calculate accuracy of prediction
for i in sorted_info:
    new_data = list()
    
    for j in attribute:
        new_data.append(encoded_data[j])
    
    new_data = np.array(new_data)

    new_data = np.transpose(new_data)
    
	# 10-Fold Cross Validation
    accuracy = 0 
    
    kf = KFold(n_splits=10, shuffle=True)
    
    for train_index, test_index in kf.split(df.Id):
        X_train = new_data[train_index]
        X_test  = new_data[test_index]
        
        Y_train = df.Label[train_index]
        Y_test = df.Label[test_index]
          
		# Use RandomForestClassifier to predict the dataset
        classifier_RF = RandomForestClassifier().fit(X_train, Y_train)
    
        yPred_RF = classifier_RF.predict(X_test)
           
        accuracy += accuracy_score(Y_test, yPred_RF)

      
    info_gain_acc.append(accuracy/10)        
    attribute.remove(i[0])

# Plot accuracy of prediction followed by number of attributes
info_gain_acc.reverse()

fig = plt.figure(figsize=(11, 7))  
plt.xticks(np.arange(0, 21, 1))
plt.gca().invert_xaxis()
plt.title("Accuracy - Number of Attributes")
plt.ylabel("Accuracy")
plt.xlabel("Number of Attributes")
plt.plot(info_gain_acc)
plt.show()
fig.savefig("Attributes-Accuracy")


# Export results to csv file
csv_out = ["Loop", "Removed Attribute", "Information Gain"]

with open('Attributes_InformationGain.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, delimiter= '\t', quoting=csv.QUOTE_NONE)
    wr.writerow(csv_out)
    for i, j in zip(sorted_info, range(20)):
        csv_out = [str(j+1), str(i[0]+1), str(i[1])]
        wr.writerow(csv_out)  
#========================================================================================================#