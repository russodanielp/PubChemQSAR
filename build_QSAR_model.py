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

def build_model(aid):


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
        print("error on aid {0}".format(aid))


    for name, clf in SKLearnModels.CLASSIFIERS:

        pipe = Pipeline(list(SKLearnModels.PREPROCESS) + [(name, clf)])
        print("=======5-fold CV on {0}=======".format(name))
        parameters = SKLearnModels.PARAMETERS[name]

        cv_search = GridSearchCV(pipe,
                           parameters,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=0)
        cv_search.fit(X.values, y.values)
        print("================================")
        print("The best parameters for {0} are :\n{1}".format(name,
                                                              cv_search.best_params_))
        print("The best accuracy score is {0}".format(cv_search.best_score_))
        # Save to pickle
        cv_search.best_estimator_



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CLI interface to build Classic ML models from PubChem')
    parser.add_argument('-a', '--aid', required=True)

    args = parser.parse_args()

    build_model(int(args.aid))