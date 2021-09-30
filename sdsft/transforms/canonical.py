"""

"""

import dssp
import numpy as np
import time

from ..common import SparseDSFT4Function, SparseWHTFunction, \
DSFT4OneHop, WHTOneHop, SparseDSFT3Function, DSFT3OneHop

import scipy
import scipy.linalg
import sdsft
    
class SparseDSFT4:
    def __init__(self, n, eps=1e-3, flag_print=False, k_max=None, flag_general=True, order='best'):
        """
            @param n: ground set size
            @param eps: |x| < eps is treated as zero
            @param flag_print: printing flag
            @param k_max: the maximum amount of frequencies maintained in all steps but the last
            @param flag_general: this toggles the filtering by a random one hop filter and is 
            required to handle arbitrary/adversary Fourier coefficients.
        """
        self.n = n
        self.k_max = k_max
        self.flag_print = flag_print
        self.eps = eps
        self.flag_general = flag_general
        if flag_general:
            self.weights = np.random.normal(0, 1, n)

    def solve_subproblem(self, s, keys_old, coefs_old, measurements_previous, M_previous):
        n = self.n
        eps = self.eps
        if self.k_max is None:
            keys_sorted = keys_old
        else:
            cards = keys_old.sum(axis=1)
            mags = -np.abs(coefs_old)
            criteria = np.zeros(len(keys_old), dtype=[('cards', '<i4'), ('coefs', '<f8')])
            criteria['cards'] = cards
            criteria['coefs'] = mags
            idx_order = np.argsort(criteria, order=('cards', 'coefs'))[:self.k_max]
            keys_sorted = keys_old[idx_order]

            M_previous = M_previous[idx_order][:, idx_order]
            measurements_previous = measurements_previous[idx_order]

        n1 = keys_sorted.shape[1]
        measurement_positions = np.zeros((keys_sorted.shape[0], n), dtype=np.int32)
        measurement_positions[:, :n1] = 1 - keys_sorted
        measurement_positions[:, n1] = 1
        measurements_new = s(measurement_positions)
        rhs = np.concatenate([measurements_new[:, np.newaxis],
                              measurements_previous[:, np.newaxis] - measurements_new[:, np.newaxis]],
                             axis=1)
        coefs = scipy.linalg.solve_triangular(M_previous, rhs, lower=True)
        n_queries = len(measurements_new)
        support_first = np.where(np.abs(coefs[:, 0]) > eps)[0]
        support_second = np.where(np.abs(coefs[:, 1]) > eps)[0]
        dim1 = len(support_first)
        dim2 = len(support_second)
        dim = len(support_first) + len(support_second)
        M = np.zeros((dim, dim), dtype=np.int32)
        M[:dim1, :dim1] = M_previous[support_first][:, support_first]
        M[dim1:, :dim1] = M_previous[support_second][:, support_first]
        M[dim1:, dim1:] = M_previous[support_second][:, support_second]
        measurements = np.concatenate([measurements_new[support_first], measurements_previous[support_second]])
        keys_first = 1 - measurement_positions[support_first][:, :n1 + 1]
        keys_second = 1 - measurement_positions[support_second][:, :n1 + 1]
        keys_second[:, -1] = 1
        keys = np.concatenate([keys_first, keys_second], axis=0)
        fourier_coefs = np.concatenate([coefs[support_first][:, 0], coefs[support_second][:, 1]])
        return fourier_coefs, keys, measurements, M, n_queries

    def transform(self, X0):
        n = self.n
        if self.flag_general:
            s = DSFT4OneHop(n, self.weights, X0)
            self.hs = s
        else:
            s = X0

        sN = s(np.zeros(n, dtype=np.bool))[0]
        M = np.ones((1, 1), dtype=np.int32)
        keys = np.zeros((1, 0), dtype=np.int32)
        fourier_coefs = np.ones(1)*sN
        measurements = np.ones(1)*sN
        partition_dict = {():sN}
            
        n_queries_total = 0
        for k in range(n):
            if len(list(partition_dict.keys())) == 0:
                keys = np.zeros((1, n), dtype=np.int32)
                fourier_coefs = np.zeros(1, dtype=np.float64)
                break
            try:
                fourier_coefs, keys, measurements, M, n_queries = self.solve_subproblem(s, keys, fourier_coefs, measurements, M)
            except ValueError as e:
                partition_dict = {}
            
            if self.flag_print:
                print('Iteration %d: queries %d dense would be %d'%(k+1, n_queries, 2**(k)))
            n_queries_total += n_queries

        if self.flag_print:
            print('Total queries: %d'%n_queries_total)
        estimate = SparseDSFT4Function(keys, fourier_coefs)

        if self.flag_general:
            estimate = s.convertCoefs(estimate)
        
        return estimate
    
class SparseDSFT3():
    def __init__(self, n, eps=1e-3, flag_print=False, k_max=None, flag_general=True,
                 order='best'):
        """
            @param n: ground set size
            @param eps: |x| < eps is treated as zero
            @param flag_print: printing flag
            @param k_max: the maximum amount of frequencies maintained in all steps but the last
            @param flag_general: this toggles the filtering by a random one hop filter and is 
            required to handle arbitrary/adversary Fourier coefficients.
        """
        self.n = n
        self.k_max = k_max
        self.flag_print = flag_print
        self.eps = eps
        self.flag_general = flag_general
        if flag_general:
            #self.weights = (1+np.random.rand(n))# + eps
            self.weights = np.random.normal(0, 1, n)
            #self.weights += np.sign(self.weights)*(eps)
            #print(self.weights)
        self.order = order


    def refine(self, s, keys_old, coefs_old, measurements_previous, M_previous):
        n = self.n
        eps = self.eps
        if self.k_max is None:
            keys_sorted = keys_old
        elif self.order == 'best':
            cards = keys_old.sum(axis=1)
            mags = -np.abs(coefs_old)
            criteria = np.zeros(len(keys_old), dtype=[('cards', '<i4'), ('coefs', '<f8')])
            criteria['cards'] = cards
            criteria['coefs'] = mags
            idx_order = np.argsort(criteria, order=('cards', 'coefs'))[:self.k_max]
            keys_sorted = keys_old[idx_order]

            M_previous = M_previous[idx_order][:, idx_order]
            measurements_previous = measurements_previous[idx_order]

        n1 = keys_sorted.shape[1]
        measurement_positions = np.ones((keys_sorted.shape[0], n), dtype=np.int32)
        measurement_positions[:, :n1] = keys_sorted
        measurement_positions[:, n1] = 0
        measurements_new = s(measurement_positions)
        rhs = np.concatenate([measurements_new[:, np.newaxis],
                              measurements_previous[:, np.newaxis] - measurements_new[:, np.newaxis]],
                             axis=1)
        coefs = scipy.linalg.solve_triangular(M_previous, rhs, lower=True)
        #coefs = np.linalg.solve(M_previous, rhs)
        n_queries = len(measurements_new)
        support_first = np.where(np.abs(coefs[:, 0]) > eps)[0]
        support_second = np.where(np.abs(coefs[:, 1]) > eps)[0]
        dim1 = len(support_first)
        dim2 = len(support_second)
        dim = len(support_first) + len(support_second)
        M = np.zeros((dim, dim), dtype=np.int32)
        M[:dim1, :dim1] = M_previous[support_first][:, support_first]
        M[dim1:, :dim1] = M_previous[support_second][:, support_first]
        M[dim1:, dim1:] = M_previous[support_second][:, support_second]
        measurements = np.concatenate([measurements_new[support_first], measurements_previous[support_second]])
        keys_first = measurement_positions[support_first][:, :n1 + 1]
        keys_second = measurement_positions[support_second][:, :n1 + 1]
        keys_second[:, -1] = 1
        keys = np.concatenate([keys_first, keys_second], axis=0)
        fourier_coefs = np.concatenate([coefs[support_first][:, 0], coefs[support_second][:, 1]])

        return fourier_coefs, keys, measurements, M, n_queries

    def transform(self, X0):
        n = self.n
        if self.flag_general:
            s = DSFT3OneHop(n, self.weights, X0)
            self.hs = s
        else:
            s = X0

        sN = s(np.ones(n, dtype=np.bool))[0]
        M = np.ones((1, 1), dtype=np.int32)#TODO:cont.
        keys = np.zeros((1, 0), dtype=np.int32)
        fourier_coefs = np.ones(1)*sN
        measurements = np.ones(1)*sN
        partition_dict = {():sN}
            
        n_queries_total = 0
        for k in range(n):
            if len(list(partition_dict.keys())) == 0:
                partition_dict[tuple(np.zeros(n, dtype=np.int32).tolist())] = 0
                break
            
            fourier_coefs, keys, measurements, M, n_queries = self.refine(s, keys, fourier_coefs, measurements, M)
            
            if self.flag_print:
                print('iteration %d: queries %d'%(k+1, n_queries))
            n_queries_total += n_queries

        if self.flag_print:
            print('total queries: %d'%(n_queries_total))
        estimate = SparseDSFT3Function(keys, fourier_coefs)
        if self.flag_general:
            estimate = s.convertCoefs(estimate)
        
        return estimate




    
    
    
    
    
    
    
    
    
    
