# -*- coding: utf-8 -*-

#libs
import logging
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import pandas as pd

# Own Modules
from sats.pysats import PySats
from hybrid_auctions.hybrid import hybrid
from hybrid_auctions.hybrid_noFR import hybrid_noFR
from hybrid_auctions.hybrid_noFR_noFA import hybrid_noFR_noFA
from mlca.mlca_value_model import ValueModel
import mlca.mlca_util as util
#%% define logger
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler) #clear existing logger
logging.basicConfig(level=logging.WARNING, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S') # log to console

#%% ENTER PARAMETERS HERE

# SATS VALUE MODEL (standard configurations):
sats_value_model = input('Select SATS Model GSVM, LSVM or MRVM: ')

if sats_value_model=='GSVM':
    V = ValueModel(name='GSVM', number_of_items=18, local_bidder_ids=[], regional_bidder_ids=list(range(0, 6)), national_bidder_ids=[6], scaler=[None])
    # NN Parameters
    epochs = [300]
    batch_size = [32]
    regularization_type = ['l2']
    regularization_R = [1e-5] # regional bidder
    learning_rate_R = [0.01]
    layer_R = [[32, 32]]
    dropout_R = [True]
    dropout_prob_R = [0.05]
    regularization_N = [1e-5] # national bidder
    learning_rate_N = [0.01]
    layer_N = [[10,10]]
    dropout_N = [True]
    dropout_prob_N = [0.05]
    # MLCA Parameters
    number_of_instances = [1] # INPUT NUMBER OF INSTANCES HERE
    start_seed = 1
    SATS_auction_instance_seeds = [list(range(start_seed, start_seed+n)) for n in number_of_instances]  # for instances in (1)
    Qround = [7]
    init_bids_and_fitted_scaler=[[None,None]]
    return_allocation=[True]
    return_payments=[True]
    calc_efficiency_per_iteration=[True]
    # MIP CPLEX parameters
    # NN-MIPS
    bigM = [2000]
    Mip_bounds_tightening = ['IA']   # False ,'IA' or 'LP'
    warm_start = [False]
    time_limit = [1800]  #in sec, 1h = 3600sec
    relative_gap = [0.001]
    integrality_tol = [1e-6]
    attempts_DNN_WDP = [5]
    # FT-MIPs
    bigM_DSPMIP = [2000]
    time_limit_DSPMIP = [1800]  #in sec, 1h = 3600sec
    relative_gap_DSPMIP = [0.001]
    integrality_tol_DSPMIP = [1e-6]
    # HYBRID parameters
    # Distribution of queries (k_i = \ell_i)
    k1=[30] # Random initial queries
    k2=[21] # MLCA allocation queries
    k3=[20] # Fourier reconstruction queries
    k4=[29] # Fourier allocation queries
    k5=[0] # MLCA Optional allocation queries
    # DSP parameters
    shift = ['shift5'] #shift3=FT3, shift4=FT4, shift5=WHT
    k_CS = [100] # Support Size returned from compressive sensing = \ell_support
    p_CS = [2000] # Superset support Size used in compressive sensing = \ell_superset
    fourier_reconstruction = [True] # parameter for wht

elif sats_value_model=='LSVM':
    V = ValueModel(name='LSVM', number_of_items=18, local_bidder_ids=[], regional_bidder_ids=list(range(1, 6)), national_bidder_ids=[0], scaler=[None])
    # NN Parameters
    epochs = [300]
    batch_size = [32]
    regularization_type = ['l2']
    regularization_R = [1e-5] # regional bidder
    learning_rate_R = [0.01]
    layer_R = [[32, 32]]
    dropout_R = [True]
    dropout_prob_R = [0.05]
    regularization_N = [1e-5] # national bidder
    learning_rate_N = [0.01]
    layer_N = [[10,10,10]]
    dropout_N = [True]
    dropout_prob_N = [0.05]
    # MLCA Parameters
    number_of_instances = [1]
    start_seed = 1
    SATS_auction_instance_seeds = [list(range(start_seed, start_seed+n)) for n in number_of_instances]  # for instances in (1)
    Qround = [6]
    init_bids_and_fitted_scaler=[[None,None]]
    return_allocation=[True]
    return_payments=[True]
    calc_efficiency_per_iteration=[True]
    # MIP CPLEX parameters
    # NN-MIPS
    bigM = [2000]
    Mip_bounds_tightening = ['IA']   # False ,'IA' or 'LP'
    warm_start = [False]
    time_limit = [1800]  #in sec, 1h = 3600sec
    relative_gap = [0.001]
    integrality_tol = [1e-6]
    attempts_DNN_WDP = [5]
    # FT-MIPs
    bigM_DSPMIP = [2000]
    time_limit_DSPMIP = [1800]  #in sec, 1h = 3600sec
    relative_gap_DSPMIP = [0.001]
    integrality_tol_DSPMIP = [1e-6]
    # HYBRID parameters
    # Distribution of queries (k_i = \ell_i)
    k1=[30] # Random initial queries
    k2=[30] # MLCA allocation queries
    k3=[10] # Fourier reconstruction queries
    k4=[30] # Fourier allocation queries
    k5=[0] # MLCA Optional allocation queries
    # DSP parameters
    shift = ['shift5'] #shift3=FT3, shift4=FT4, shift5=WHT
    k_CS = [100] # Support Size returned from compressive sensing = \ell_support
    p_CS = [2000] # Superset support Size used in compressive sensing = \ell_superset
    fourier_reconstruction = [True] # parameter for wht

elif sats_value_model=='MRVM':
    V = ValueModel(name='MRVM', number_of_items=98, local_bidder_ids=[0, 1, 2], regional_bidder_ids=[3, 4, 5, 6], national_bidder_ids=[7, 8, 9], scaler=[MinMaxScaler(feature_range=(0, 500))])
    # NN Parameters
    epochs = [300]
    batch_size = [32]
    regularization_type = ['l2']
    regularization_L = [1e-5] # local bidders
    learning_rate_L = [0.01]
    layer_L = [[16, 16]]
    dropout_L = [True]
    dropout_prob_L = [0.05]
    regularization_R = [1e-5] # regional bidders
    learning_rate_R = [0.01]
    layer_R = [[16, 16]]
    dropout_R = [True]
    dropout_prob_R = [0.05]
    regularization_N = [1e-5] # national bidders
    learning_rate_N = [0.01]
    layer_N = [[16, 16]]
    dropout_N = [True]
    dropout_prob_N = [0.05]
    # MLCA Parameters
    number_of_instances = [1]
    start_seed = 1
    SATS_auction_instance_seeds = [list(range(start_seed, start_seed+n)) for n in number_of_instances]  # for instances in (1)
    Qround = [4]
    init_bids_and_fitted_scaler=[[None,None]]
    return_allocation=[True]
    return_payments=[True]
    calc_efficiency_per_iteration=[True]
    # MIP CPLEX parameters
    # NN-MIPS
    bigM = [2000]
    Mip_bounds_tightening = ['IA']   # False ,'IA' or 'LP'
    warm_start = [False]
    time_limit = [300]  #in sec, 1h = 3600sec
    relative_gap = [0.01]
    integrality_tol = [1e-6]
    attempts_DNN_WDP = [5]
    # FT-MIPs
    bigM_DSPMIP = [2000]
    time_limit_DSPMIP = [60]  #in sec, 1h = 3600sec
    relative_gap_DSPMIP = [0.001]
    integrality_tol_DSPMIP = [1e-6]
    # HYBRID parameters
    # Distribution of queries (k_i = \ell_i)
    k1=[30] # Random initial queries
    k2=[220] # MLCA allocation queries
    k3=[0] # Fourier reconstruction queries
    k4=[250] # Fourier allocation queries
    k5=[0] # MLCA Optional allocation queries
    # DSP parameters
    shift = ['shift5'] #shift3=FT3, shift4=FT4, shift5=WHT
    k_CS = [500] # Support Size returned from compressive sensing = \ell_support
    p_CS = [2000] # Superset support Size used in compressive sensing = \ell_superset
    fourier_reconstruction = [False] # parameter for wht
else:
    raise NotImplementedError('Selected SATS value model not implemented.')

if not fourier_reconstruction[0]:
    print('TOTAL QUERIES:', sum(k1+k2+k4+k5))
else:
    print('TOTAL QUERIES:', sum(k1+k2+k3+k4+k5))
# %% CREATE CONFIG FILES

# (i) Neural Network Parameters
NN_keys = ['regularization', 'learning_rate', 'architecture', 'dropout', 'dropout_prob','epochs','batch_size','regularization_type']
L_NN_parameters = None
if V.name == 'MRVM':
    L_NN_parameters = list(product(regularization_L, learning_rate_L, layer_L, dropout_L, dropout_prob_L, epochs, batch_size,regularization_type))
    L_NN_parameters = [OrderedDict(zip(NN_keys, x)) for x in L_NN_parameters]
R_NN_parameters = list(product(regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R, epochs, batch_size,regularization_type))
R_NN_parameters = [OrderedDict(zip(NN_keys, x)) for x in R_NN_parameters]
N_NN_parameters = list(product(regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N, epochs, batch_size,regularization_type))
N_NN_parameters = [OrderedDict(zip(NN_keys, x)) for x in N_NN_parameters]

if L_NN_parameters is not None:
    bidder_keys = ['Local', 'Regional', 'National']
    NN_parameters = list(product(L_NN_parameters, R_NN_parameters, N_NN_parameters))
    NN_parameters = [OrderedDict(zip(bidder_keys, x)) for x in NN_parameters]
else:
    bidder_keys = ['Regional', 'National']
    NN_parameters = list(product(R_NN_parameters, N_NN_parameters))
    NN_parameters = [OrderedDict(zip(bidder_keys, x)) for x in NN_parameters]
NN_parameters = V.parameters_to_bidder_id(NN_parameters)

# (ii) Mixed Integer Program Parameters
MIP_keys =['bigM', 'mip_bounds_tightening', 'warm_start', 'time_limit', 'relative_gap','integrality_tol','attempts_DNN_WDP','bigM_DSPMIP','time_limit_DSPMIP','relative_gap_DSPMIP','integrality_tol_DSPMIP']
MIP_parameters = list(product(bigM, Mip_bounds_tightening, warm_start, time_limit, relative_gap, integrality_tol, attempts_DNN_WDP,bigM_DSPMIP,time_limit_DSPMIP,relative_gap_DSPMIP,integrality_tol_DSPMIP))
MIP_parameters = [OrderedDict(zip(MIP_keys, x)) for x in MIP_parameters]

# (iii) Set all parameters
Parameter_keys = ['SATS_domain_name','SATS_auction_instance_seeds', 'Qround','NN_parameters', 'MIP_parameters','scaler', 'number_of_instances_in_config',
                  'init_bids_and_fitted_scaler', 'return_allocation', 'return_payments', 'calc_efficiency_per_iteration','k1','k2','k3','k4','k5','k_CS','p_CS','shift','fourier_reconstruction']
CONFIGS = [OrderedDict(zip(Parameter_keys, x)) for x in list(product([V.name], SATS_auction_instance_seeds, Qround, NN_parameters, MIP_parameters,
                                                                     V.scaler, number_of_instances,init_bids_and_fitted_scaler,return_allocation,return_payments,
                                                                     calc_efficiency_per_iteration, k1, k2, k3, k4, k5, k_CS, p_CS, shift, fourier_reconstruction))]
j = 0
for x in CONFIGS:
        print('CONFIG {}'.format(j))
        util.pretty_print_dict(x, printing=True)
        print()
        j = j + 1

CONFIGS_REP = util.helper_f(CONFIGS)[0]
# %% START MECHANISMS
mechanism = input('SELECT MECHANISM FROM: Hybrid=1, Hybrid-FR=2, Hybrid-FR-FA=3 via inputting integer from {1,2,3}.')

EFFICIENCY = {}
TOTAL_TIME_ELAPSED = {}
EFFICIENCY_PER_ITER = {}
REVENUE = {}
DISTRIBUTION_SCW = {}
if mechanism == '1':
    logging.warning('Running Hybrid Mechanism')
    for configdict in CONFIGS_REP:
        seed = 'Seed {}'.format(configdict['SATS_auction_instance_seed'])
        # RUN HYBRID
        R = hybrid(configdict)
        #EFFICIENCY
        EFFICIENCY[seed] = R[1]['MLCA Efficiency']
        #TOTAL TIME ELAPSED
        TOTAL_TIME_ELAPSED[seed] = R[1]['Statistics']['Total Time Elapsed'] # (d,h,m,s)
        #äEFFICIENCY PER ITERATION
        EFFICIENCY_PER_ITER[seed] = R[1]['Statistics']['Efficiency per Iteration']
        #REVENUE
        REVENUE[seed] = R[1]['Statistics']['Relative Revenue']
        # DISTRIBUTION OF SCW
        if sats_value_model == 'LSVM':
            SATS_auction_instance = PySats.getInstance().create_lsvm(seed=configdict['SATS_auction_instance_seed'], isLegacyLSVM=True)
            National_index = ['Bidder_0']
            Regional_index = ['Bidder_1', 'Bidder_2', 'Bidder_3', 'Bidder_4', 'Bidder_5']
        if sats_value_model == 'GSVM':
            SATS_auction_instance = PySats.getInstance().create_gsvm(seed=configdict['SATS_auction_instance_seed'], isLegacyGSVM=True)
            National_index = ['Bidder_6']
            Regional_index = ['Bidder_0', 'Bidder_1', 'Bidder_2', 'Bidder_3', 'Bidder_4', 'Bidder_5']
        if sats_value_model == 'MRVM':
            SATS_auction_instance = PySats.getInstance().create_mrvm(seed=configdict['SATS_auction_instance_seed'])
            National_index = ['Bidder_7', 'Bidder_8', 'Bidder_9']
            Regional_index = ['Bidder_3', 'Bidder_4', 'Bidder_5', 'Bidder_6']
            Local_index = ['Bidder_0', 'Bidder_1', 'Bidder_2']
        _,SATS_SCW = SATS_auction_instance.get_efficient_allocation()
        MECHANISM_ALLOCATION = pd.DataFrame.from_dict(R[1]['MLCA Allocation']).transpose()
        Local_percentages = 0
        if sats_value_model == 'MRVM':
            Local_percentages = MECHANISM_ALLOCATION['value'][Local_index].sum()/SATS_SCW
        Regional_percentages = MECHANISM_ALLOCATION['value'][Regional_index].sum()/SATS_SCW
        National_percentages = MECHANISM_ALLOCATION['value'][National_index].sum()/SATS_SCW
        DISTRIBUTION_SCW[seed] = {'Local Bidders':Local_percentages, 'Regional Bidders':Regional_percentages, 'National Bidders': National_percentages}
        # PRINT STORED RESULTS
        logging.warning('\n')
        logging.warning('FINAL RESULTS')
        logging.warning('--------------------------------------------------------------')
        logging.warning('Efficiency         :   {} %'.format(round(100*EFFICIENCY[seed], 2)))
        logging.warning('Time elapsed       :   {} h'.format(round(TOTAL_TIME_ELAPSED[seed][0]*24+TOTAL_TIME_ELAPSED[seed][1]+TOTAL_TIME_ELAPSED[seed][2]/60+TOTAL_TIME_ELAPSED[seed][3]/3600,2)))
        logging.warning('Relative Revenue   :   {} %'.format(round(100*REVENUE[seed],2), '%'))
        logging.warning('Distribution of SCW')
        if sats_value_model == 'MRVM':
            logging.warning('Local Bidders  :   {} %'.format(round(Local_percentages*100, 2)))
        logging.warning('Regional Bidders   :   {} %'.format(round(Regional_percentages*100, 2)))
        logging.warning('National Bidders   :   {} %'.format(round(National_percentages*100, 2)))
        logging.warning('Efficiency per Iteration')
        for k,v in EFFICIENCY_PER_ITER[seed].items():
            logging.warning(k)
            for k2,v2 in v.items():
                logging.warning('Iteration {}        :   {} %'.format(k2,round(v2*100,2)))
        logging.warning('--------------------------------------------------------------')
        logging.warning('\n')
        del R

elif mechanism == '2':
    logging.warning('Running Hybrid-FR Mechanism')
    for configdict in CONFIGS_REP:
        seed = 'Seed {}'.format(configdict['SATS_auction_instance_seed'])
        #RUN HYBRID-FR
        R = hybrid_noFR(configdict)
        #EFFICIENCY
        EFFICIENCY[seed] = R[1]['MLCA Efficiency']
        #TOTAL TIME ELAPSED
        TOTAL_TIME_ELAPSED[seed] = R[1]['Statistics']['Total Time Elapsed']
        #äEFFICIENCY PER ITERATION
        EFFICIENCY_PER_ITER[seed] = R[1]['Statistics']['Efficiency per Iteration']
        #REVENUE
        REVENUE[seed] = R[1]['Statistics']['Relative Revenue']
        # DISTRIBUTION OF SCW
        if sats_value_model == 'LSVM':
            SATS_auction_instance = PySats.getInstance().create_lsvm(seed=configdict['SATS_auction_instance_seed'], isLegacyLSVM=True)
            National_index = ['Bidder_0']
            Regional_index = ['Bidder_1', 'Bidder_2', 'Bidder_3', 'Bidder_4', 'Bidder_5']
        if sats_value_model == 'GSVM':
            SATS_auction_instance = PySats.getInstance().create_gsvm(seed=configdict['SATS_auction_instance_seed'], isLegacyGSVM=True)
            National_index = ['Bidder_6']
            Regional_index = ['Bidder_0', 'Bidder_1', 'Bidder_2', 'Bidder_3', 'Bidder_4', 'Bidder_5']
        if sats_value_model == 'MRVM':
            SATS_auction_instance = PySats.getInstance().create_mrvm(seed=configdict['SATS_auction_instance_seed'])
            National_index = ['Bidder_7', 'Bidder_8', 'Bidder_9']
            Regional_index = ['Bidder_3', 'Bidder_4', 'Bidder_5', 'Bidder_6']
            Local_index = ['Bidder_0', 'Bidder_1', 'Bidder_2']
        _,SATS_SCW = SATS_auction_instance.get_efficient_allocation()
        MECHANISM_ALLOCATION = pd.DataFrame.from_dict(R[1]['MLCA Allocation']).transpose()
        Local_percentages = 0
        if sats_value_model == 'MRVM':
            Local_percentages = MECHANISM_ALLOCATION['value'][Local_index].sum()/SATS_SCW
        Regional_percentages = MECHANISM_ALLOCATION['value'][Regional_index].sum()/SATS_SCW
        National_percentages = MECHANISM_ALLOCATION['value'][National_index].sum()/SATS_SCW
        DISTRIBUTION_SCW[seed] = {'Local Bidders':Local_percentages, 'Regional Bidders':Regional_percentages, 'National Bidders': National_percentages}
        # PRINT STORED RESULTS
        logging.warning('\n')
        logging.warning('FINAL RESULTS')
        logging.warning('--------------------------------------------------------------')
        logging.warning('Efficiency         :   {} %'.format(round(100*EFFICIENCY[seed], 2)))
        logging.warning('Time elapsed       :   {} h'.format(round(TOTAL_TIME_ELAPSED[seed][0]*24+TOTAL_TIME_ELAPSED[seed][1]+TOTAL_TIME_ELAPSED[seed][2]/60+TOTAL_TIME_ELAPSED[seed][3]/3600,2)))
        logging.warning('Relative Revenue   :   {} %'.format(round(100*REVENUE[seed],2), '%'))
        logging.warning('Distribution of SCW')
        if sats_value_model == 'MRVM':
            logging.warning('Local Bidders  :   {} %'.format(round(Local_percentages*100, 2)))
        logging.warning('Regional Bidders   :   {} %'.format(round(Regional_percentages*100, 2)))
        logging.warning('National Bidders   :   {} %'.format(round(National_percentages*100, 2)))
        logging.warning('Efficiency per Iteration')
        for k,v in EFFICIENCY_PER_ITER[seed].items():
            logging.warning(k)
            for k2,v2 in v.items():
                logging.warning('Iteration {}        :   {} %'.format(k2,round(v2*100,2)))
        logging.warning('--------------------------------------------------------------')
        logging.warning('\n')
        del R

elif mechanism=='3':
    logging.warning('Running Hybrid-FR-FA Mechanism')
    for configdict in CONFIGS_REP:
        seed = 'Seed {}'.format(configdict['SATS_auction_instance_seed'])
        #RUN HYBRID-FR-FA
        R = hybrid_noFR_noFA(configdict)
        #EFFICIENCY
        EFFICIENCY[seed] = R[1]['MLCA Efficiency']
        #TOTAL TIME ELAPSED
        TOTAL_TIME_ELAPSED[seed] = R[1]['Statistics']['Total Time Elapsed']
        #äEFFICIENCY PER ITERATION
        EFFICIENCY_PER_ITER[seed] = R[1]['Statistics']['Efficiency per Iteration']
        #REVENUE
        REVENUE[seed] = R[1]['Statistics']['Relative Revenue']
        # DISTRIBUTION OF SCW
        if sats_value_model == 'LSVM':
            SATS_auction_instance = PySats.getInstance().create_lsvm(seed=configdict['SATS_auction_instance_seed'], isLegacyLSVM=True)
            National_index = ['Bidder_0']
            Regional_index = ['Bidder_1', 'Bidder_2', 'Bidder_3', 'Bidder_4', 'Bidder_5']
        if sats_value_model == 'GSVM':
            SATS_auction_instance = PySats.getInstance().create_gsvm(seed=configdict['SATS_auction_instance_seed'], isLegacyGSVM=True)
            National_index = ['Bidder_6']
            Regional_index = ['Bidder_0', 'Bidder_1', 'Bidder_2', 'Bidder_3', 'Bidder_4', 'Bidder_5']
        if sats_value_model == 'MRVM':
            SATS_auction_instance = PySats.getInstance().create_mrvm(seed=configdict['SATS_auction_instance_seed'])
            National_index = ['Bidder_7', 'Bidder_8', 'Bidder_9']
            Regional_index = ['Bidder_3', 'Bidder_4', 'Bidder_5', 'Bidder_6']
            Local_index = ['Bidder_0', 'Bidder_1', 'Bidder_2']
        _,SATS_SCW = SATS_auction_instance.get_efficient_allocation()
        MECHANISM_ALLOCATION = pd.DataFrame.from_dict(R[1]['MLCA Allocation']).transpose()
        Local_percentages = 0
        if sats_value_model == 'MRVM':
            Local_percentages = MECHANISM_ALLOCATION['value'][Local_index].sum()/SATS_SCW
        Regional_percentages = MECHANISM_ALLOCATION['value'][Regional_index].sum()/SATS_SCW
        National_percentages = MECHANISM_ALLOCATION['value'][National_index].sum()/SATS_SCW
        DISTRIBUTION_SCW[seed] = {'Local Bidders':Local_percentages, 'Regional Bidders':Regional_percentages, 'National Bidders': National_percentages}
        # PRINT STORED RESULTS
        logging.warning('\n')
        logging.warning('FINAL RESULTS')
        logging.warning('--------------------------------------------------------------')
        logging.warning('Efficiency         :   {} %'.format(round(100*EFFICIENCY[seed], 2)))
        logging.warning('Time elapsed       :   {} h'.format(round(TOTAL_TIME_ELAPSED[seed][0]*24+TOTAL_TIME_ELAPSED[seed][1]+TOTAL_TIME_ELAPSED[seed][2]/60+TOTAL_TIME_ELAPSED[seed][3]/3600,2)))
        logging.warning('Relative Revenue   :   {} %'.format(round(100*REVENUE[seed],2), '%'))
        logging.warning('Distribution of SCW')
        if sats_value_model == 'MRVM':
            logging.warning('Local Bidders      :   {} %'.format(round(Local_percentages*100, 2)))
        logging.warning('Regional Bidders   :   {} %'.format(round(Regional_percentages*100, 2)))
        logging.warning('National Bidders   :   {} %'.format(round(National_percentages*100, 2)))
        logging.warning('Efficiency per Iteration')
        for k,v in EFFICIENCY_PER_ITER[seed].items():
            logging.warning(k)
            for k2,v2 in v.items():
                logging.warning('Iteration {}        :   {} %'.format(k2,round(v2*100,2)))
        logging.warning('--------------------------------------------------------------')
        logging.warning('\n')
        del R
else:
    raise NotImplementedError('Mechanism {} not implemented yet.'.format(mechanism))