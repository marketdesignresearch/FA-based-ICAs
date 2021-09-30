# -*- coding: utf-8 -*-

# Packages
from datetime import datetime
import logging
import numpy as np
from collections import OrderedDict
from copy import deepcopy
# Own modules
import sdsft
import dssp
from mlca.mlca import mlca_mechanism
from mlca.mlca_economies import MLCA_Economies
from mlca.mlca_util import timediff_d_h_m_s
from hybrid_auctions.dsp_auction_functions import key_to_int, fit_NNs,fit_dsp_models, update_bids, dsp_allocational_queries
import functools
import time
#%%

def wrapnn(x, nn):
    return nn.predict(x.astype(np.float64)).T[0].astype(np.float64)

def hybrid(configdict):

    start = datetime.now()
    EFFICIENCY_PER_ITER={}
    MIP_TIMINGS = {}

    shift = configdict['shift']
    k1 = configdict['k1']
    k2 = configdict['k2']
    k3 = configdict['k3']
    k4 = configdict['k4']
    k5 = configdict['k5']
    k_CS = configdict['k_CS']
    p_CS = configdict['p_CS']
    SATS_domain_name = configdict['SATS_domain_name']
    NN_parameters = configdict['NN_parameters']

    # (1-2) MLCA: k1+k2 Allocational Queries
    logging.warning('(1-2) MLCA: k1:%s Init. Random Queries and k2:%s Allocation Queries',k1,k2)
    #---------------------------------------------------------------------------------------------------------------------------------------
    configdict['Qinit']=k1
    configdict['Qmax']=configdict['Qinit']+k2
    mlca1 = mlca_mechanism(configdict=configdict)
    ELICITED_BIDS = deepcopy(mlca1[1]['Elicited Bids'])
    EFFICIENCY_PER_ITER['MLCA_k1+k2'] = deepcopy(mlca1[1]['Statistics']['Efficiency per Iteration'])
    MIP_TIMINGS['MLCA_k1+k2'] = deepcopy(mlca1[1]['Statistics']['Elapsed Times of MIPs'])
    SATS_auction_instance = mlca1[1]['SATS_AUCTION_INSTANCE']
    bidder_ids = list(SATS_auction_instance.get_bidder_ids())
    #---------------------------------------------------------------------------------------------------------------------------------------


    # (3) DSP: FF-Analysis of NNs
    logging.warning('')
    logging.warning('(3) FT-Analysis of NNs using FT: %s', shift)
    #---------------------------------------------------------------------------------------------------------------------------------------
    TF_MODELS = fit_NNs(elicited_bids=ELICITED_BIDS,
                        NN_parameters=NN_parameters,
                        fitted_scaler=mlca1[1]['Fitted Scaler'])

    NN_SFS = {}
    for key, nn in TF_MODELS.items():
        function = functools.partial(wrapnn, nn=nn)
        NN_SFS[key] = sdsft.WrapSetFunction(function)

    BIDDER_SFS = OrderedDict()
    for bidder_id in bidder_ids:
        BIDDER_SFS['Bidder_%d'%bidder_id] = sdsft.Bidder(SATS_auction_instance, bidder_id)

    SUPPORTS = OrderedDict()
    COEFS = OrderedDict()
    if SATS_domain_name == 'GSVM' or SATS_domain_name == 'LSVM':
        n_goods = 18
        indicator_matrix = np.asarray([dssp.int2indicator(A, n_goods) for A in range(2**n_goods)])
        cardinalities = np.sum(indicator_matrix, axis=1)
        for key, nn in NN_SFS.items():
            s = nn(indicator_matrix)

            if shift == 'shift4':
                s_hat = dssp.fdsft4(s)
                weights = -2**((n_goods - cardinalities)/2)
                support = np.asarray([dssp.int2indicator(B, n_goods).astype(np.int32) for B in np.argsort(weights*np.abs(s_hat))[:k3]])
                coefs = s_hat[np.argsort(weights*np.abs(s_hat))[:k3]]
            elif shift == 'shift5':
                s_hat = dssp.fwht(s)
                support = np.asarray([dssp.int2indicator(B, n_goods).astype(np.int32) for B in np.argsort(-np.abs(s_hat))[:p_CS]])
                coefs = s_hat[np.argsort(-np.abs(s_hat))[:p_CS]]
            elif shift == 'shift3':
                s_hat = dssp.fdsft3(s)
                s_hat = (-1)**cardinalities * s_hat #make it compatible with the sparse dsft3 function implementation
                weights = -2**((n_goods - cardinalities)/2)
                support = np.asarray([dssp.int2indicator(B, n_goods).astype(np.int32) for B in np.argsort(weights*np.abs(s_hat))[:k3]])
                coefs = s_hat[np.argsort(weights*np.abs(s_hat))[:k3]]
            else:
                raise NotImplementedError

            SUPPORTS[key] = support
            COEFS[key] = coefs

    elif SATS_domain_name == 'MRVM':
        n_goods = 98

        if shift == 'shift4':
            ft = sdsft.SparseDSFT4(n_goods, eps=1e-3, flag_general=False, flag_print=False, k_max=1000)
        elif shift == 'shift5':
            ft = sdsft.RWHT(n_goods, p_CS)
        elif shift == 'shift3':
            ft = sdsft.SparseDSFT3(n_goods, eps=1e-3, flag_general=False, flag_print=False, k_max=1000)
        else:
            raise NotImplementedError

        for key, nn in NN_SFS.items():
            estimate = ft.transform(nn)
            #estimate = sdsft.SparseWHTFunction(np.asarray([]), np.asarray([]), normalization=False)

            if len(estimate.coefs) == 0:
                # if the robust wht failed, we use low frequencies instead...
                freqs = [np.zeros(n_goods, dtype=np.int32)]
                for i in range(n_goods):
                    freq = np.zeros(n_goods, dtype=np.int32)
                    freq[i] = 1
                    freqs += [freq]

                for i in range(n_goods-1):
                    for j in range(i+1, n_goods):
                        freq = np.zeros(n_goods, dtype=np.int32)
                        freq[i] = 1
                        freq[j] = 1
                        freqs += [freq]
                freqs = np.asarray(freqs)
                coefs = np.zeros(len(freqs), dtype=np.float64)
                estimate.freqs = freqs
                estimate.coefs = coefs

            cardinalities = estimate.freqs.sum(axis=1)
            support = []
            coefs = []
            if shift == 'shift4':
                weights = -2**((n_goods - cardinalities)/2)
                for idx in np.argsort(weights*np.abs(estimate.coefs))[:k3]:
                    support += [estimate.freqs[idx].tolist()]
                    coefs += [estimate.coefs[idx]]
            elif shift == 'shift5':
                for idx in np.argsort(-np.abs(estimate.coefs))[:p_CS]:
                    support += [estimate.freqs[idx].tolist()]
                    coefs += [estimate.coefs[idx]]
            elif shift == 'shift3':
                weights = -2**((n_goods - cardinalities)/2)
                for idx in np.argsort(weights*np.abs(estimate.coefs))[:k3]:
                    support += [estimate.freqs[idx].tolist()]
                    coefs += [estimate.coefs[idx]]
            else:
                raise NotImplementedError

            support = np.asarray(support)
            SUPPORTS[key] = support
            COEFS[key] = np.asarray(coefs)
    else:
        raise NotImplementedError
    #---------------------------------------------------------------------------------------------------------------------------------------


    # (4-5) DSP: Sampling Theorem & k3 Reconstructing Queries
    logging.warning('')
    logging.warning('(4-5) FT: k3:%s Fourier Reconstructing Queries using Shift: %s',k3, shift)
    logging.warning('Fourier Reconstruction Queries: %s', configdict.get('fourier_reconstruction', True))
    #---------------------------------------------------------------------------------------------------------------------------------------
    NEW_BIDS = OrderedDict()

    if shift == 'shift4':
        for key, support in SUPPORTS.items():
            measurement_positions = 1 - support
            measurements = BIDDER_SFS[key](measurement_positions)
            if SATS_domain_name == 'MRVM':
                measurements = mlca1[1]['Fitted Scaler'].transform(measurements[:,np.newaxis])[:,0]
            NEW_BIDS[key] = [measurement_positions, measurements]
        ELICITED_BIDS = update_bids(old=ELICITED_BIDS, new=NEW_BIDS)
    elif shift == 'shift5':
        if configdict.get('fourier_reconstruction', True):
            for key, support in SUPPORTS.items():
                measurement_positions = 1 - support[:k3]
                measurements = BIDDER_SFS[key](measurement_positions)
                if SATS_domain_name == 'MRVM':
                    measurements = mlca1[1]['Fitted Scaler'].transform(measurements[:,np.newaxis])[:,0]
                NEW_BIDS[key] = [measurement_positions, measurements]
            ELICITED_BIDS = update_bids(old=ELICITED_BIDS, new=NEW_BIDS)
    elif shift == 'shift3':
        for key, support in SUPPORTS.items():
            measurement_positions = support
            measurements = BIDDER_SFS[key](measurement_positions)
            if SATS_domain_name == 'MRVM':
                measurements = mlca1[1]['Fitted Scaler'].transform(measurements[:,np.newaxis])[:,0]
            NEW_BIDS[key] = [measurement_positions, measurements]
        ELICITED_BIDS = update_bids(old=ELICITED_BIDS, new=NEW_BIDS)
    else:
        raise NotImplementedError

    # Calculate efficient allocation given current elicited bids
    E = MLCA_Economies(SATS_auction_instance=SATS_auction_instance,
                              SATS_auction_instance_seed=configdict['SATS_auction_instance_seed'],
                              Qinit=0, Qmax=0, Qround=0, scaler=False)
    E.set_initial_bids(initial_bids=ELICITED_BIDS, fitted_scaler=mlca1[1]['Fitted Scaler']) # (*) use self defined inital bids | Line 1
    if configdict['calc_efficiency_per_iteration']: E.calculate_efficiency_per_iteration()
    EFFICIENCY_PER_ITER['FT_Reconstruction_k3'] = E.efficiency_per_iteration
    del E

    time1= time.time()
    DSP_MODELS = fit_dsp_models(ELICITED_BIDS, SUPPORTS, shift=shift, k_CS = k_CS)
    time2=time.time()
    logging.warning('Time for fitting FT models: %f s'%(time2-time1))

    #---------------------------------------------------------------------------------------------------------------------------------------


    # (6) DSP: k4 Allocational Queries
    logging.warning('')
    logging.warning('(6) FT: k4:%s Fourier Allocation Queries using Shift %s',k4, shift)
    #---------------------------------------------------------------------------------------------------------------------------------------
    mip_timings = []
    for i in range(k4):
        #(6.1)
        #NEW GSVM specific constraints
        if (SATS_auction_instance.get_model_name()=='GSVM' and not SATS_auction_instance.isLegacy):
            GSVM_specific_constraints = True
            national_circle_complement = list(set(SATS_auction_instance.get_good_ids())-set(SATS_auction_instance.get_goods_of_interest(bidder_id=6)))
            logging.warning('########## ATTENTION ##########')
            logging.warning('GSVM specific constraints: %s', GSVM_specific_constraints)
            logging.warning('###############################\n')
        else:
            GSVM_specific_constraints = False
            national_circle_complement = None

        logging.warning('Fourier allocation query no. %s',i+1)
        dsp_queries, mip_runtime = dsp_allocational_queries(models=DSP_MODELS,
                                                            elicited_bids=ELICITED_BIDS,
                                                            mip_parameters=configdict['MIP_parameters'],
                                                            GSVM_specific_constraints=GSVM_specific_constraints,
                                                            national_circle_complement=national_circle_complement)
        mip_timings += mip_runtime
        NEW_BIDS = {}
        for k, v in dsp_queries.items():
            measurement_positions = v.reshape(1,-1)
            measurements = np.ones((1,1))*SATS_auction_instance.calculate_value(bidder_id=key_to_int(k),
                                         goods_vector=v)
            if SATS_domain_name == 'MRVM':
                measurements = mlca1[1]['Fitted Scaler'].transform(measurements)[:,0]
            else:
                measurements = measurements[:,0]
            NEW_BIDS[k] = [measurement_positions, measurements]

        ELICITED_BIDS = update_bids(old=ELICITED_BIDS,new=NEW_BIDS)
        #(6.2)
        time1= time.time()
        DSP_MODELS = fit_dsp_models(ELICITED_BIDS, SUPPORTS, shift=shift, k_CS = k_CS)
        time2=time.time()
        logging.warning('Time for fitting FT models: %f s'%(time2-time1))


    MIP_TIMINGS['DSP_Allocational_k4'] = mip_timings
    # Calculate efficient allocation given current elicited bids
    E = MLCA_Economies(SATS_auction_instance=SATS_auction_instance,
                              SATS_auction_instance_seed=configdict['SATS_auction_instance_seed'],
                              Qinit=0, Qmax=0, Qround=0, scaler=False)
    E.set_initial_bids(initial_bids=ELICITED_BIDS, fitted_scaler=mlca1[1]['Fitted Scaler']) # (*) use self defined inital bids | Line 1
    if configdict['calc_efficiency_per_iteration']: E.calculate_efficiency_per_iteration()
    EFFICIENCY_PER_ITER['FT_Allocation_k4'] = E.efficiency_per_iteration
    del E
    #---------------------------------------------------------------------------------------------------------------------------------------


    # (7-8) MLCA: optional rounds
    logging.warning('')
    logging.warning('(7-8) MLCA optional rounds: k5:%s Allocation Queries',k5)
    #---------------------------------------------------------------------------------------------------------------------------------------
    configdict_new = deepcopy(configdict)
    if configdict_new.get('fourier_reconstruction', True):
        configdict_new['Qmax']=(k1+k2+k3+k4)+k5  # Qinit is set automatically via shape of given bids
    else:
        configdict_new['Qmax']=(k1+k2+k4)+k5  # Qinit is set automatically via shape of given bids
    configdict_new['init_bids_and_fitted_scaler']=[ELICITED_BIDS, mlca1[1]['Fitted Scaler']]
    mlca2 = mlca_mechanism(configdict=configdict_new)
    EFFICIENCY_PER_ITER['MLCA_k5'] = deepcopy(mlca2[1]['Statistics']['Efficiency per Iteration'])
    MIP_TIMINGS['MLCA_k5'] = deepcopy(mlca2[1]['Statistics']['Elapsed Times of MIPs'])
    mlca2[1]['Statistics']['Efficiency per Iteration'] = EFFICIENCY_PER_ITER
    mlca2[1]['Statistics']['Elapsed Times of MIPs'] = MIP_TIMINGS
    del mlca2[1]['SATS_AUCTION_INSTANCE']  #cannot be pickled somehow
    #---------------------------------------------------------------------------------------------------------------------------------------
    end = datetime.now()
    total_time_elapsed = timediff_d_h_m_s(end-start)
    mlca2[1]['Statistics']['Total Time Elapsed'] = total_time_elapsed
    return(mlca2)