import logging
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)


class PubChemDataSetCleaner:
    """ pipeline class for cleaner """

    def __init__(self, steps=[]):
        self._steps = steps

    def addProcess(self, process):
        return PubChemDataSetCleaner(self._steps+[process])

    def run(self, ds):
        ds_copy = ds.copy()
        ds_copy['rdkit'] = [Chem.MolFromSmiles(smi) for smi in ds_copy.SMILES]
        for step in self._steps:
            ds_copy = step.runStep(ds_copy)
        return ds_copy

class PubChemDataSetStep:
    def runStep(self, ds):
        raise NotImplementedError("Please implement this method")

class StructureCleaner(PubChemDataSetStep):
    """ class that drops any compound whose structure does not result in a rdkit mol """

    def runStep(self, ds):
        # converts rdkit mols back to smiles to see if it results in actual smiles string
        ds_copy = ds.copy()

        mols = []
        for idx, mol in ds_copy.rdkit.iteritems():
            if not mol:
                mols.append(None)
            else:
                try:
                    AllChem.Compute2DCoords(mol)
                    mols.append(mol)
                except:
                    mols.append(None)
        ds_copy['rdkit_checker'] = [Chem.MolToSmiles(mol) if mol else None for mol in ds_copy.rdkit]
        not_none = ~ds_copy.rdkit_checker.isnull()
        return ds[not_none]

class ActivityBalancer(PubChemDataSetStep):
    """ class that balances active:inactive ratio """

    def runStep(self, ds):

        actives, inactives = self.splitter(ds)
        # balance dataset by taking equalizing actives and inactives
        # to the smaller of the two
        if len(actives) < len(inactives):
            inactives = self.selector(inactives, len(actives))
        else:
            actives = self.selector(actives, len(inactives))
        return pd.concat([actives, inactives])

    def splitter(self, ds):
        """ split dataset into actives and inactives """
        actives = ds[ds.Activity == 1]
        inactives = ds[ds.Activity == 0]
        return actives, inactives

    def selector(self, ds, n):
        """ Randomize the dataset, then selects the first n elements """
        import random
        random_numbers = list(range(len(ds)))
        random.shuffle(random_numbers)
        random_numbers = random_numbers[:n]
        return ds.iloc[random_numbers, :]


