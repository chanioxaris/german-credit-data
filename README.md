## Overview

This is an analysis and classification of german credit data (more information at [this pdf](https://github.com/chanioxaris/GermanCreditData-Classification/blob/master/docs/german-1.pdf)). Three classifiers tested, [Support Vector Machines (SVM)](http://scikit-learn.org/stable/modules/svm.html), [Random Forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html), to select the most efficient for our data. The code implemented in Python 3.6 using [scikit-learn](http://scikit-learn.org/stable/) library.


## Data visualization

For each attribute i used two different plots to represent their data spreading depending on data kind . 
- [Histogram](https://en.wikipedia.org/wiki/Histogram): for categorical data
- [Box plot](https://en.wikipedia.org/wiki/Box_plot): for numerical data


![Attribute1](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute1.png)   |  ![Attribute2](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute2.png) 
:-------------------------:|:-------------------------:
![Attribute3](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute3.png)   |  ![Attribute4](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute4.png) 
![Attribute5](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute5.png)   |  ![Attribute6](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute6.png) 
![Attribute7](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute7.png)   |  ![Attribute8](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute8.png) 
![Attribute9](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute9.png)   |  ![Attribute10](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute10.png) 
![Attribute11](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute11.png)   |  ![Attribute12](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute12.png) 
![Attribute13](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute13.png)   |  ![Attribute14](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute14.png) 
![Attribute15](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute15.png)   |  ![Attribute16](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute16.png) 
![Attribute17](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute17.png)   |  ![Attribute18](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute18.png) 
![Attribute19](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute19.png)   |  ![Attribute20](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attribute20.png) 





## Classifier accuracy

I used [10-Fold Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) method for each of three classifiers to measure their accuracy on multiple parts of train dataset. The results are the follows.

| Statistic Measure | Naive Bayes | Random Forests |   SVM   | 
| :---------------: | :---------: | :-----------: | :-----: | 
|      Accuracy     |   0.71250   |    0.73875    | 0.70125 |   


The [Random Forests classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) performs better than others two and i am gonna use it for my final prediction.

## Information gain

In general terms, the [information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition) is the change in [information entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) H from a prior state to a state that takes some information as given:
`IG(T,a) = H(T) - H(T|a)`


Information gain for each attribute sorted ascending is represented in the following table.

| Attribute Number |  Information Gain | 
| :--------------: | :---------------: | 
|        18        | 0.000129665701928 |  
|        11        | 0.000220571349274 | 
|        19        | 0.001202862591080 |   
|        16        | 0.002395770112590 |   
|        17        | 0.002940316631290 |  
|        10        | 0.005674399790160 |  
|        14        | 0.007041506325140 | 
|        08        | 0.007330500076830 |   
|        20        | 0.007704386546440 |   
|        15        | 0.011618886823700 |
|        09        | 0.012746841156200 |  
|        13        | 0.013412980533000 | 
|        07        | 0.014547865230200 |   
|        12        | 0.014905530877300 |   
|        05        | 0.018461146132300 |
|        06        | 0.022198966052400 |
|        04        | 0.026897452033100 |  
|        02        | 0.032963429423100 | 
|        03        | 0.037889406221500 |   
|        01        | 0.093827963023500 |   


The next step, is to loop through all attributes and removing one each time from dataset based on the above table. I calculated again the accuracy to find the ideal number of attributes that achieve the best accuracy.

![Accuracy](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/img/Attributes-Accuracy.png)


As we can see in the above plot, we achieve the best accuracy after removing the first eight attributes with the lowest [information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition). We are gonna use the rest twelve remaining attributes to predict our final dataset.

## Prediction 

Final, we perform our prediction using [Random Forests classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) trained by `dataset_train.tsv` and using only those attributes which resulted the highest prediction accuracy, as we saw at the above plot. Results exported to [Predictions.csv](https://github.com/chanioxaris/GermanCreditData-Mining/blob/master/output/Predictions.csv) file.


## Usage

For windows based systems `python information_gain.py` and `python predict.py`

For linux bases systems `py information_gain.py` and `py predict.py`
