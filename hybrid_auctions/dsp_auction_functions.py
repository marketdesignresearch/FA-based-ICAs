# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file implements helper functions for various FT-based auction formats.
"""

# Libs
import numpy as np
import itertools
import logging
import re
import time
import scipy
from copy import deepcopy
from collections import OrderedDict
# Own Modules
from hybrid_auctions.dsp_mip import DspMip
from hybrid_auctions.wdp import WDP
from mlca.mlca_nn import MLCA_NN
import sdsft
from sklearn.linear_model import lars_path
# %%
def key_to_int(key):
    return(int(re.findall(r'\d+', key)[0]))
# %%
def predict_dsp(model, x):
    if model['dsp_type'] == 'shift5':
        return(np.dot(model['v_hat'],(-1)**np.dot(model['W'],x)))
    if model['dsp_type'] == 'shift4':
        k = len(model['v_hat'])
        return(np.dot(model['v_hat'],np.maximum(np.zeros(k), np.ones(k)-np.dot(model['W'], x))))
#%% required format for old and new: {'Bidder_i':[bundles,values]}_i
def update_bids(old, new):
    tmp = deepcopy(old)
    for bidder in old.keys():
        logging.info('SHAPE NEW: [X,Y]:=[%s, %s]',new[bidder][0].shape, new[bidder][1].shape)
        # update bundles
        tmp[bidder][0] = np.append(old[bidder][0],new[bidder][0], axis=0)
        # update values
        tmp[bidder][1] = np.append(old[bidder][1],new[bidder][1], axis=0) # update values
        logging.info('CHECK Uniqueness of updated elicited bids for %s:', bidder)
        check = len(np.unique(tmp[bidder][0], axis=0))==len(tmp[bidder][0])
        if check:
            logging.info('UNIQUE')
        else:
            logging.info('NOT UNIQUE\n')
        logging.info('UPDATED SHAPE: %s\n',tmp[bidder][0].shape)
    return(tmp)
#%%
def fit_NNs(elicited_bids,NN_parameters,fitted_scaler,):
    models = OrderedDict()
    for bidder in elicited_bids:
        logging.info(bidder)
        bids = elicited_bids[bidder]
        start = time.time()
        nn_model = MLCA_NN(X_train=bids[0], Y_train=bids[1], scaler=fitted_scaler)  # instantiate class
        nn_model.initialize_model(model_parameters=NN_parameters[bidder])  # initialize model
        nn_model.fit(epochs=NN_parameters[bidder]['epochs'], batch_size=NN_parameters[bidder]['batch_size'],  # fit model to data
                           X_valid=None, Y_valid=None)
        end = time.time()
        logging.info('Time for ' + bidder + ': %s sec\n', round(end-start))
        models[bidder] = nn_model.model
    return(models)
# %%
def check_bundle_contained(bundle, bidder, elicited_bids):
    if np.any(np.equal(elicited_bids[bidder][0],bundle).all(axis=1)):
        logging.info('Argmax bundle ALREADY ELICITED from {}'.format(bidder))
        return(True)
    return(False)
# %%
def dsp_allocational_queries(models, elicited_bids, mip_parameters, GSVM_specific_constraints=False, national_circle_complement=None):
    mip_timings = []
    X = DspMip(models = models)
    X.initialize_mip(M=mip_parameters['bigM_DSPMIP'],
                     feasibility_equality=False,
                     GSVM_specific_constraints=GSVM_specific_constraints,
                     national_circle_complement=national_circle_complement)  # for now enter here 'feasibility_equality' for CPLEX.

    X.solve_mip(log_output=False,
                time_limit=mip_parameters['time_limit_DSPMIP'],
                mip_relative_gap=mip_parameters['relative_gap_DSPMIP'],
                mip_start=None,
                integrality_tol=mip_parameters['integrality_tol_DSPMIP']
                )

    mip_timings.append(X.soltime)

    dsp_allocational_queries = {k:None for k in X.optimal_allocation.keys()}

    for bidder in elicited_bids.keys():
        CHECK = check_bundle_contained(bundle=X.optimal_allocation[bidder], bidder=bidder, elicited_bids=elicited_bids)
        if CHECK:
            XX = DspMip(models = models)
            XX.initialize_mip(M=mip_parameters['bigM_DSPMIP'],
                              feasibility_equality=False,
                              bidder_specific_constraints={bidder:elicited_bids[bidder][0]},
                              GSVM_specific_constraints=GSVM_specific_constraints,
                              national_circle_complement=national_circle_complement)  # for now enter here 'feasibility_equality' for CPLEX.
            XX.solve_mip(log_output=False,
                         time_limit=mip_parameters['time_limit_DSPMIP'],
                         mip_relative_gap=mip_parameters['relative_gap_DSPMIP'],
                         mip_start=None,
                         integrality_tol=mip_parameters['integrality_tol_DSPMIP']
                         )
            dsp_allocational_queries[bidder] = XX.optimal_allocation[bidder]
            mip_timings.append(XX.soltime)
            del XX
        else:
            dsp_allocational_queries[bidder] = X.optimal_allocation[bidder]

    return(dsp_allocational_queries, mip_timings)
# %%
def fit_dsp_models(ELICITED_BIDS, SUPPORTS, shift='shift4', k_CS=None):
    DSP_MODELS = dict()
    keys = ['v_hat', 'W', 'dsp_type']
    if shift == 'shift4':
        for key in ELICITED_BIDS.keys():
            measurement_positions = ELICITED_BIDS[key][0]
            measurements = ELICITED_BIDS[key][1]
            support = SUPPORTS[key]
            design_matrix = (measurement_positions.dot(support.T) == 0).astype(np.float64)
            fourier_coefs, residues, rank, singular_values = scipy.linalg.lstsq(design_matrix, measurements)
            DSP_MODELS[key] = dict(zip(keys, [fourier_coefs, support, shift]))
    elif shift == 'shift3':
        for key in ELICITED_BIDS.keys():
            measurement_positions = ELICITED_BIDS[key][0]
            measurements = ELICITED_BIDS[key][1]
            support = SUPPORTS[key]
            design_matrix = ((1 - measurement_positions).dot(support.T) == 0).astype(np.float64)
            fourier_coefs, residues, rank, singular_values = scipy.linalg.lstsq(design_matrix, measurements)
            DSP_MODELS[key] = dict(zip(keys, [fourier_coefs, support, shift]))
    elif shift == 'shift5':
        bidders = []
        for key in ELICITED_BIDS.keys():
            measurement_positions = ELICITED_BIDS[key][0]
            measurements = ELICITED_BIDS[key][1]
            support = SUPPORTS[key]
            if k_CS is None:
                k_CS = len(measurements)
            A = (-1)**measurement_positions.dot(support.T).astype(np.float64)
            alphas, active, coefs = lars_path(A, measurements, max_iter=k_CS, method='lasso')
            freqs = support[active]
            coefs = coefs[:, -1][active]
            est = sdsft.SparseWHTFunction(freqs, coefs, normalization=False)
            bidders += [est]

        DSP_MODELS = sdsft.bidders2dict(bidders, 5)

    return DSP_MODELS