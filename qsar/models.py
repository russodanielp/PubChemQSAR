from sklearn.base import ClassifierMixin, BaseEstimator

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

import logging

log = logging.getLogger(__name__)

class SKLearnModels:

    PREPROCESS = (
        ('scaler', MinMaxScaler(feature_range=[0, 1])),
        ('variance', VarianceThreshold()),
        ('selectKbest', SelectKBest(chi2, k=10))
    )

    # TODO whats the deal with calibrated CV
    CLASSIFIERS = (
        ('Random Forest', RandomForestClassifier()),
        ('Support Vector Classification', SVC()),
        ('Naive Bayes', BernoulliNB()),
        ('Logistic Regression', LogisticRegression())
        #('kNN', KNeighborsClassifier())

    )

    PARAMETERS = {
        'Random Forest':{
            'Random Forest__n_estimators': [5, 10, 25],
            'Random Forest__max_features': ['auto', 'sqrt', 'log2', None],
            'Random Forest__max_depth': [5],
            'Random Forest__criterion': ['gini', 'entropy'],
            'Random Forest__n_jobs': [-1],
            'Random Forest__class_weight': ['balanced'],

        },
        'kNN':{
            'kNN__n_neighbors': [1, 2, 3, 4, 5],
            'kNN__weights': ['uniform', 'distance'],
            'kNN__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'kNN__leaf_size': np.linspace(1,100, 10).tolist(),
            'kNN__metric': ['euclidean', 'manhattan',
                       'chebyshev', 'minkowski',
                       'seuclidean', 'mahalanobis'],
            'kNN__p': [1, 2],
            'kNN__n_jobs': [-1]
        },
        'Support Vector Classification':{
            'Support Vector Classification__class_weight': ['balanced'],
            'Support Vector Classification__kernel': ['rbf'],
            'Support Vector Classification__gamma': [1e-2, 1e-3],
            'Support Vector Classification__C': [1, 10]
        },
        'Logistic Regression':{
            'Logistic Regression__C': list(np.logspace(-5, -1, 5)),
        },
        'Naive Bayes':{
            'Naive Bayes__alpha': [1]
        }
    }

    DESCRIPTORS = (
        'Rdkit'
    )

def CreateQSARClassifcationModel(classifier):
    """ function that generates a QSAR classifcation model from a sklearn model """

    class QSARClassificationModel(classifier):
        """ Class that inherits properties of sklearn classifiers """

        def __init__(self):
            params = classifier().get_params()
            classifier.__init__(self, **params)

        def __str__(self):
            return "<QSAR Model>"

        def __repr__(self):
            return self.__str__()

    # returns an instance of an QSARClassificationModel
    # not sure if it is more pythonic to find a way
    # to return an uninstantiated object
    return QSARClassificationModel


# #
# # class ConsensusModel:
# #     """ container class for all QSARClassifcationModels """
# #
# #     rf_rdkit = QSARRFClassification('rdkit')
# #     svm_rdkit = QSARSVClassification('rdkit')
# #     nb_rdkit = QSARNBClassification('rdkit')
# #
# #
# #     def __str__(self):
# #         s = 'Consensus model consisting of {0} models\n'.format(len(self._models))
# #         for model in self._models:
# #             s += str(model)+'\n'
# #         return s
# #
# #     @staticmethod
# #     def fit(model, X, y):
# #         return model.fit(X, y)