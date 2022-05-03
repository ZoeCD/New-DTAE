import numpy as np


class Feature:
    def __init__(self, name, index, values):
        self.name = name
        self.index = index
        self.values = values
        self.missing_value_objects = []
        self.type = 'Numerical' if np.issubdtype(np.array(values).dtype, np.number) else 'Nominal'
        self.encoded_value_names = []
        self.imputation_value = None
        self.encoder = None
        self.valid_feature_index = None

    def get_name(self):
        return self.name

    def get_values(self):
        return self.values

    def get_missing_value_indexes(self):
        return self.missing_value_objects

    def set_missing_value_objects(self, indexes):
        self.missing_value_objects = indexes

    def set_encoded_value_names(self, names):
        self.encoded_value_names = names

    def set_imputation_value(self, value):
        self.imputation_value =  value

    def get_imputation_value(self):
        return self.imputation_value