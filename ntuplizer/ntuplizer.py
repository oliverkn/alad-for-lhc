from abc import ABC, abstractmethod

import uproot as ur
import numpy as np


class Ntuplizer:
    def __init__(self):
        self.selectors = []
        self.quantities = []

    def register_selector(self, s):
        assert isinstance(s, AbstractSelectorModule)
        self.selectors.append(s)

    def register_quantity_module(self, q):
        assert isinstance(q, AbstractQuantityModule)
        self.quantities.append(q)

    def convert(self, input_file, target_file):
        f = ur.open(input_file)
        e = f['Events']

        n_events = e.__len__()
        n_quantities = 0

        # provide values
        values = {}

        for q in self.quantities:
            n_quantities += q.get_size()

            for key in q.get_keys():
                if key in values:
                    continue
                values[key] = e.array(key)

        # compute selection mask
        mask = np.ones(n_events, dtype=bool)
        for selector in self.selectors:
            mask *= selector.select(values)

        # apply selection mask
        for key, value in values.items():
            values[key] = value[mask]

        n_events = np.sum(mask)
        result = np.empty(shape=(n_events, n_quantities))

        # compute all quantities and add to result
        j = 0
        for q in self.quantities:
            val = q.compute(values)
            s = q.get_size()
            result[:, j:j + s] = val
            j += s

        np.save(target_file, result)


class AbstractQuantityModule(ABC):
    @abstractmethod
    def compute(self, values): pass

    @abstractmethod
    def get_keys(self): pass

    @abstractmethod
    def get_size(self): pass

    @abstractmethod
    def get_names(self): pass


class AbstractSelectorModule(ABC):
    @abstractmethod
    def select(self, values): pass

    @abstractmethod
    def get_keys(self): pass

    @abstractmethod
    def get_name(self): pass
