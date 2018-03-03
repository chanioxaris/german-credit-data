## Overview

This is an analysis and classification of german credit data (more information at docs/german-1.pdf). Three classifiers tested, [Support Vector Machines (SVM)](http://scikit-learn.org/stable/modules/svm.html), [Random Forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html), to predict multiple parts of dataset using 10-Fold Cross Validation and measure theis accuracy before actual prediction. The code is written on Python 3.6 and with the help of [scikit-learn](http://scikit-learn.org/stable/) library.


| Statistic Measure | Naive Bayes | Random Forest |   SVM   | 
| :---------------: | :---------: | :-----------: | :-----: | 
|      Accuracy     |   0.71250   |    0.73875    | 0.70125 |   



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


## Usage

For windows based systems `python information_gain.py` and `python predict.py`

For linux bases systems `py information_gain.py` and `py predict.py`
