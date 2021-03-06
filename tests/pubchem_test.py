from rdkit import Chem

from qsar.pubchem import PubChemDataSet


class TestPubChemDataSet:

    def setup(self):
        pass

    def teardown(self):
        pass

    def test_PubChemDataSet_creation(self):
        """ testing PubChemDataSet creation """

        # some cids from PubChem AID 1
        active_compounds = [11969872, 390525, 394646]
        inactive_compounds = [1018, 4775, 219294]
        ds = PubChemDataSet(1)
        df = ds.get_compounds()

        assert all(df.loc[active_compounds, 'Activity'] == 1)
        assert all(df.loc[inactive_compounds, 'Activity'] == 0)


    def test_PubChemDataSet_load(self):
        """ testing PubChemDataSet load """

        # some smiles from PubChem AID 1
        smiles = ['COC1=CC(=C2C(=C1)OC(=CC2=O)C3(C=CC(=O)C=C3)O)O',
                  'C1=CC=C(C=C1)CCCC(=O)O']

        df = PubChemDataSet(1).load()
        for smi in smiles:
            print(smi in df.SMILES.tolist())
            assert smi in df.SMILES.tolist()

    def test_PubChemDataSet_clean_load(self):
        """ testing a clean loaded dataset """

        df = PubChemDataSet(1).clean_load()
        assert len((df.Activity[df.Activity == 1])) == len((df.Activity[df.Activity == 0]))
        assert None not in [Chem.MolToSmiles(mol) if mol else None for mol in df.rdkit]