# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file stores helper functions used across the files in this project.

"""

# Libs
import numpy as np
import random
import re
import logging
from collections import OrderedDict
import copy

#%% replicate configs with seeds_instance

def helper_f(CONFIGS):
    tmp1 = []
    for c in CONFIGS:
        tmp2 = []
        for seed in c['SATS_auction_instance_seeds']:
            y = copy.deepcopy(c)
            y['SATS_auction_instance_seed'] = seed
            tmp2.append(y)
            del y
        tmp1.append(tmp2)
    return(tmp1)
# %%
def pretty_print_dict(D, printing=True):
    text = []
    for key, value in D.items():
        if key in ['NN_parameters','MIP_parameters']:
            if printing:
                print(key, ': ')
            text.append(key + ': \n')
            for k, v in value.items():
                if printing:
                    print(k+': ', v)
                text.append(k + ': ' + str(v) + '\n')
        else:
            if printing:
                print(key, ':  ', value)
            text.append(key + ':  ' + str(value) + '\n')
    return(''.join(text))
# %%
def timediff_d_h_m_s(td):
    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return -(td.days), -int(td.seconds/3600), -int(td.seconds/60)%60, -(td.seconds%60)
    return td.days, int(td.seconds/3600), int(td.seconds/60)%60, td.seconds%60

# %% Tranforms bidder_key to integer bidder_id
# key = valid bidder_key (string), e.g. 'Bidder_0'
def key_to_int(key):
        return(int(re.findall(r'\d+', key)[0]))
# %% PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS for MLCA MECHANISM
# THIS METHOD USES TRUE UNIFORM SAMPLING!
# SATS_auction_instance = single instance of a value model
# number_initial_bids = number of initial bids
# bidder_ids = bidder ids in this value model (int)
# scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set
# seed = seed for random initial bids

def initial_bids_mlca_unif(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None, seed=None):
    initial_bids = OrderedDict()

    # seed determines bidder_seeds for all bidders, e.g. seed=1 and 3 bidders generates bidder_seeds=[3,4,5]
    n_bidders = len(bidder_names)
    if seed is not None:
        bidder_seeds = range(seed*n_bidders,(seed+1)*n_bidders)
    else:
        bidder_seeds = [None]*n_bidders

    i = 0
    for bidder in bidder_names:
        logging.debug('Random Bids for: %s using seed %s', bidder, bidder_seeds[i])

        #Deprecated version
        #D = unif_random_bids(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)

        #New version
        #updated to new uniform sampling method from SATS, which incorporates bidder specific restrictions in GSVM
        #i.e., for regional bidders only buzndles of up to size 4 are sampled, for national bidders only bundles that
        #contain items from the national circle are sampled.
        #Remark: SATS does not ensure that bundles are unique, this needs to be taken care exogenously.
        D = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=key_to_int(bidder),
                                                                     number_of_bids=number_initial_bids,
                                                                     seed=bidder_seeds[i]))
        # get unique ones if sampled equal bundles
        unique_indices = np.unique(D[:,:-1],return_index=True,axis=0)[1]
        seed_additional_bundle = None if bidder_seeds[i] is None else 10**6*bidder_seeds[i]
        while len(unique_indices) != number_initial_bids:
            tmp = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=key_to_int(bidder),
                                                                      number_of_bids=1,
                                                                      seed=seed_additional_bundle))
            D = np.vstack((D, tmp))
            unique_indices = np.sort(np.unique(D[:,:-1],return_index=True,axis=0)[1])
            seed_additional_bundle +=1

        D = D[unique_indices,:]
        logging.debug('Shape %s', D.shape)

        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        X = X.astype(int)  #convert bundles to numpy integers
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]
        i += 1

    if scaler is not None:
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))

    return(initial_bids, scaler)

# %% RANDOM ADDITIONAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS for MLCA + RANDOM MECHANISM in FT PROJECT
# THIS METHOD USES TRUE UNIFORM SAMPLING!
# SATS_auction_instance = single instance of a value model
# number_random_bids = number of random bids
# bidder_ids = bidder ids in this value model (int)
# fitted_scaler = scale the y values across all bidders, already fitted on a prespecified training set
# seed = seed for random bids

def random_bids_mlca_unif(SATS_auction_instance, number_random_bids, bidder_names, fitted_scaler=None, seed=None):
    random_bids = OrderedDict()

    # seed determines bidder_seeds for all bidders, e.g. seed=1 and 3 bidders generates bidder_seeds=[3,4,5]
    n_bidders = len(bidder_names)
    if seed is not None:
        bidder_seeds = range(seed*n_bidders,(seed+1)*n_bidders)
    else:
        bidder_seeds = [None]*n_bidders

    i = 0
    for bidder in bidder_names:
        logging.debug('Random Bids for: %s using seed %s', bidder, bidder_seeds[i])

        #DEPRECATED VERSION
        #D = unif_random_bids(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_random_bids)

        #NEW VERSION
        #updated to new uniform sampling method from SATS, which incorporates bidder specific restrictions in GSVM
        #i.e., for regional bidders only buzndles of up to size 4 are sampled, for national bidders only bundles that
        #contain items from the national circle are sampled.
        D = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=key_to_int(bidder),
                                                                     number_of_bids=number_random_bids,
                                                                     seed=bidder_seeds[i]))
        # get unique ones if sampled equal bundles
        unique_indices = np.unique(D[:,:-1],return_index=True,axis=0)[1]
        seed_additional_bundle = None if bidder_seeds[i] is None else 10**6*bidder_seeds[i]
        while len(unique_indices) != number_random_bids:
            tmp = np.asarray(SATS_auction_instance.get_uniform_random_bids(bidder_id=key_to_int(bidder),
                                                                      number_of_bids=1,
                                                                      seed=seed_additional_bundle))
            D = np.vstack((D, tmp))
            unique_indices = np.sort(np.unique(D[:,:-1],return_index=True,axis=0)[1])
            seed_additional_bundle +=1

        D = D[unique_indices,:]
        logging.debug('Shape %s', D.shape)
        i += 1

        X = D[:, :-1]
        X = X.astype(int)  # convert bundles to numpy integers
        Y = D[:, -1]
        random_bids[bidder] = [X, Y]

    if fitted_scaler is not None:
        minI = int(round(fitted_scaler.data_min_[0]*fitted_scaler.scale_[0]))
        maxI = int(round(fitted_scaler.data_max_[0]*fitted_scaler.scale_[0]))
        logging.debug('Random Bids scaled by: %s to the interval [%s,%s]',round(fitted_scaler.scale_[0],8),minI,maxI)
        random_bids = OrderedDict(list((key, [value[0], fitted_scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in random_bids.items()))

    return(random_bids)


# %% This function formates the solution of the winner determination problem (WDP) given elicited bids.
# Mip = A solved DOcplex instance.
# elicited_bids = the set of elicited bids for each bidder corresponding to the WDP.
# bidder_names = bidder names (string, e.g., 'Bidder_1')
# fitted_scaler = the fitted scaler used in the valuation model.


def format_solution_mip_new(Mip, elicited_bids, bidder_names, fitted_scaler):
    tmp = {'good_ids': [], 'value': 0}
    Z = OrderedDict()
    for bidder_name in bidder_names:
        Z[bidder_name] = tmp
    S = Mip.solution.as_dict()
    for key in list(S.keys()):
        index = [int(x) for x in re.findall(r'\d+', key)]
        bundle = elicited_bids[index[0]][index[1], :-1]
        value = elicited_bids[index[0]][index[1], -1]
        if fitted_scaler is not None:
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug(value)
            logging.debug('WDP values for allocation scaled by: 1/%s',round(fitted_scaler.scale_[0],8))
            value = float(fitted_scaler.inverse_transform([[value]]))
            logging.debug(value)
            logging.debug('---------------------------------------------')
        bidder = bidder_names[index[0]]
        Z[bidder] = {'good_ids': list(np.where(bundle == 1)[0]), 'value': value}
    return(Z)