# -*- coding: utf-8 -*-

# Packages
from datetime import datetime
import logging
from copy import deepcopy
# Own modules
from mlca.mlca import mlca_mechanism
from mlca.mlca_economies import MLCA_Economies
from mlca.mlca_util import timediff_d_h_m_s, random_bids_mlca_unif
from hybrid_auctions.dsp_auction_functions import  update_bids
#%%

def hybrid_noFR_noFA(configdict):

    start = datetime.now()
    EFFICIENCY_PER_ITER={}

    k1 = configdict['k1']
    k2 = configdict['k2']
    k3 = configdict['k3'] # here these are the random queries
    k4 = configdict['k4'] # here these are the random queries
    k5 = configdict['k5']

    # (1-2) MLCA: k1+k2 Allocational Queries
    logging.warning('')
    logging.warning('(1-2) MLCA: k1:%s Init. Random Queries and k2:%s Allocation Queries',k1,k2)
    #---------------------------------------------------------------------------------------------------------------------------------------
    configdict['Qinit']=k1
    configdict['Qmax']=configdict['Qinit']+k2
    mlca1 = mlca_mechanism(configdict=configdict)
    ELICITED_BIDS = deepcopy(mlca1[1]['Elicited Bids'])
    EFFICIENCY_PER_ITER['MLCA_k1+k2'] = deepcopy(mlca1[1]['Statistics']['Efficiency per Iteration'])
    SATS_auction_instance = mlca1[1]['SATS_AUCTION_INSTANCE']
    bidder_names = list(ELICITED_BIDS.keys())
    #---------------------------------------------------------------------------------------------------------------------------------------

    # (3-6) RANDOM: k3+k4 random Queries
    #---------------------------------------------------------------------------------------------------------------------------------------
    logging.warning('')
    logging.warning('(3-6) RANDOM: k3+k4: %s Random Queries', k3+k4)
    NEW_BIDS = random_bids_mlca_unif(SATS_auction_instance=SATS_auction_instance,
                                     number_random_bids=k3+k4, bidder_names=bidder_names,
                                     fitted_scaler=mlca1[1]['Fitted Scaler'], seed=None)
    ELICITED_BIDS = update_bids(old=ELICITED_BIDS, new=NEW_BIDS)

    # Calculate efficient allocation given current elicited bids
    E = MLCA_Economies(SATS_auction_instance=SATS_auction_instance,
                              SATS_auction_instance_seed=configdict['SATS_auction_instance_seed'],
                              Qinit=0, Qmax=0, Qround=0, scaler=False)
    E.set_initial_bids(initial_bids=ELICITED_BIDS, fitted_scaler=mlca1[1]['Fitted Scaler']) # (*) use self defined inital bids | Line 1
    if configdict['calc_efficiency_per_iteration']: E.calculate_efficiency_per_iteration()
    EFFICIENCY_PER_ITER['RANDOM_k3+k4'] = E.efficiency_per_iteration
    del E
    #---------------------------------------------------------------------------------------------------------------------------------------


    # (7-8) MLCA: optional rounds
    logging.warning('')
    logging.warning('(7-8) MLCA optional rounds: k5:%s Allocation Queries',k5)
    #---------------------------------------------------------------------------------------------------------------------------------------
    configdict_new = deepcopy(configdict)
    configdict_new['Qmax']=(k1+k2+k3+k4)+k5  # Qinit is set automatically via shape of given bids
    configdict_new['init_bids_and_fitted_scaler']=[ELICITED_BIDS, mlca1[1]['Fitted Scaler']]
    mlca2 = mlca_mechanism(configdict=configdict_new)
    EFFICIENCY_PER_ITER['MLCA_k5'] = deepcopy(mlca2[1]['Statistics']['Efficiency per Iteration'])
    mlca2[1]['Statistics']['Efficiency per Iteration'] = EFFICIENCY_PER_ITER
    del mlca2[1]['SATS_AUCTION_INSTANCE']  #cannot be pickled somehow
    #---------------------------------------------------------------------------------------------------------------------------------------
    end = datetime.now()
    total_time_elapsed = timediff_d_h_m_s(end-start)
    mlca2[1]['Statistics']['Total Time Elapsed'] = total_time_elapsed
    return(mlca2)