import pandas as pd
from abc import abstractmethod
from rdkit import Chem
import os
import logging
import getpass

log = logging.getLogger(__name__)

class Profile:

    def __init__(self, profile, activity, smiles):
        self.profile = profile
        self.activity = activity
        self.smiles = smiles

    def get_subprofile(self, aids: list, drop_null=False):
        """ returns a subset of the profile specified by aids as a Profile object

        aids: aids to keep
        drop_null: return compounds with no responses
        """
        for aid in aids:
            if aid not in self.profile.columns:
                raise Exception("{0} is not in the profile, "
                                "AIDs in profile are {1}".format(aid, self.columns.tolist()))
        sub_profile = self.profile.copy()[aids]
        if drop_null:
            profile = self.remove_nulls(sub_profile)
            return Profile(profile, self.activity[profile.index], self.smiles[profile.index])
        return Profile(sub_profile, self.activity, self.smiles)

    def remove_cmps(self, cmps: list):
        """ return a subset of the Profile minus the specified compounds

        cmps: cmps to remove
        drop_null: return compounds with no responses
        """
        for cmp in cmps:
            if cmp not in self.profile.index:
                raise Exception("{0} is not in the profile, "
                                "AIDs in profile are {1}".format(cmp, self.index.tolist()))
        indices = ~self.profile.index.isin(cmps)
        return Profile(self.profile[indices], self.activity[indices], self.smiles[indices])

    def as_ds(self):
        """ rerturns a DataFrame of activity and smiles in columns """
        return pd.concat([self.activity, self.smiles], axis=1)

    def get_nulls(self):
        """ return Profile object with compounds that have no responses in all aids
            This will come in handy for getting compounds with no activity to make predictions on

        """
        profile = self.profile[(self.profile == 0).all(1)]
        return Profile(profile, self.activity[profile.index], self.smiles[profile.index])

    def remove_nulls(self, profile):
        """ remove compounds with no response in any aid """
        return profile[(profile != 0).any(1)]

    def __add__(self, other):
        if not isinstance(other, Profile):
            raise Exception("{0} is not of type Profile".format(other))

        if self.profile.shape[0] != other.profile.shape[0]:
            raise Exception("Can not add profiles of "
                            "shape {0} and {1}.  Profiles must"
                            "be the same and equal along the index."
                            "".format(self.profile.shape[0], other.profile.shape[0]))


        if any(self.profile.index != other.profile.index):
            raise Exception("Can not add profiles. Profiles must"
                            "be the same and equal along the index."
                            "".format(self.profile.shape[0], other.profile.shape[0]))
        return Profile(pd.concat([self.profile, other.profile], axis=1), self.activity, self.smiles)

