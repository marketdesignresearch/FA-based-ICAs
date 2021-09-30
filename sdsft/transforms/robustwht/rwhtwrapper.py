#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:18:44 2020

"""

from .sparseWHT_robust import SWHTRobust
import numpy as np
from ...common import SparseWHTFunction

class SetFunctionVector():
    def __init__(self, s, n):
        """
        @param s: SetFunction
        """
        self.s = s
        self.n = n
        self.shape = tuple(n*[2])
        
    def __getitem__(self, indicator):
        return self.s(np.asarray(indicator))[0]

    def __call__(self, indicator):
        return self.s(indicator)
    
class RWHT():
    def __init__(self, n, K):
        self.n = n
        self.K = K
        self.wht = SWHTRobust(n, K)
    
    def transform(self, s):
        res = self.wht.run(SetFunctionVector(s, self.n))
        freqs = np.asarray(list(res.keys())).astype(np.int32)
        coefs = np.asarray(list(res.values())).astype(np.float64)
        return SparseWHTFunction(freqs, coefs, normalization=False)
    
        