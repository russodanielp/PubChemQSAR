# PubChemQSAR



PubChemQSAR is a Python program that can build Quantitative Structure Activity
Relationship (QSAR) models from any given PubChem Assay Identifier (AID).  Given the PubChem AID, PubChemQSAR grab
 compound assay responses and structural information.  It will the clean the data, balance the active to inactive ration,
 and train a classification model using 5-fold cross validation on any of 5 machine learning algorithms (Random Forest,
 Support Vector Machine, Naive Bayes, Logistic Regression, and kNN).


 Example usage:

 ```bash
 python build_QSAR_model.py -a 101 -m rf
 ```

 Example output:

 ```
=======building model for aid 101=======
=======4102 compounds: 2033 active, 2069 inactive=======
=======5-fold CV on Random Forest=======
================================
The best parameters for Random Forest are :
criterion: entropy
max_depth: 5
n_jobs: -1
max_features: None
class_weight: balanced
n_estimators: 25
The best accuracy score is 72.11%
 ```


