import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from qsar.descriptors import PubChemDataSetDescriptors
from qsar.models import SKLearnModels
from qsar.pubchem import PubChemDataSet

# suppress warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def build_model(aid, model):


    try:
        ds = PubChemDataSet(aid).clean_load()
        y = ds.Activity
        X = PubChemDataSetDescriptors(ds).load_rdkit()

        # TODO: put this into a cleaner step
        # remove null values
        y = y[X.notnull().all(1)]
        X = X[X.notnull().all(1)]

        # TODO: put this into a cleaner step
        # remove null values
        y = y[~np.isinf(X.values).any(1)]
        X = X[~np.isinf(X.values).any(1)]
        print("=======building model for aid {0}=======".format(aid))
        print("======={0} compounds: {1} active, {2} inactive=======".format(y.shape[0],
                                                                             (y == 1).sum(),
                                                                             (y == 0).sum()))
    except:
        raise Exception("error on aid {0}".format(aid))


    pipe = Pipeline(list(SKLearnModels.PREPROCESS) + [(name, clf) for name, clf in SKLearnModels.CLASSIFIERS if name == model])
    print("=======5-fold CV on {0}=======".format(model))
    parameters = SKLearnModels.PARAMETERS[model]

    cv_search = GridSearchCV(pipe,
                       parameters,
                       cv=5,
                       scoring='accuracy',
                       n_jobs=-1,
                       verbose=0)
    cv_search.fit(X.values, y.values)
    print("================================")
    print("The best parameters for {0} are :".format(model))
    for param, val in cv_search.best_params_.items():
        print("{}: {}".format(param.split('__')[1], val))
    print("The best accuracy score is {0:.2f}%".format(cv_search.best_score_ * 100))
    # Save to pickle
    cv_search.best_estimator_



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CLI interface to build Classic ML models from PubChem')
    parser.add_argument('-a', '--aid', required=True)
    parser.add_argument('-m', '--model', required=True)

    model_mapper = {'rf':'Random Forest',
                    'svm':'Support Vector Classification',
                    'nb':'Naive Bayes',
                    'lr':'Logistic Regression',
                    'knn':'kNN'}

    args = parser.parse_args()

    build_model(int(args.aid), model_mapper[args.model])