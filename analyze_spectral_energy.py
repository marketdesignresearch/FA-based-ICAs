# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:29:36 2021

@author: jakob
"""

# libs
import numpy as np
import random
import os
import pickle
from datetime import datetime
import time
import json

# own modules
from plot_utils import  plot_spectrum, print_elapsed_time
import dssp
from sats.pysats import PySats

if __name__ == '__main__':


    # %% Define Set Function v: 2^M -> R_+ for spectral analysis

    # EXAMPLE LSVM NATIONAL BIDDER:
    # ---------------------------------------------------------------------------------------
    # For this example we use as set function v the national bidder from the SATS domain LSVM.

    # SELECT **********************
    #load_flag = False
    sats_value_model = 'LSVM'
    sats_bidder_type = 'national'
    seed = 1
    # *****************************


    SATS_auction_instance = PySats.getInstance().create_lsvm(seed=seed, isLegacyLSVM=True)
    m = len(SATS_auction_instance.get_good_ids()) # number of items (size pof ground set); worksup to m=32
    bidder_type_to_id_mapping = {'regional':[1,2,3,4,5],'national':[0]}
    bidder_id = random.sample(bidder_type_to_id_mapping[sats_bidder_type],1)[0]

    # create set function v, which gets as input a indicator list of size m, representing the set, and outputs a real number.
    def v(x):
        return SATS_auction_instance.calculate_value(bidder_id=bidder_id,
                                                     goods_vector=x)

    # test
    print('\n\nCheck set function v:')
    print(f'v((0,...,0))) = {v([0]*m):6.2f}')
    print(f'v((1,...,1))) = {v([1]*m):.2f}')
    # ---------------------------------------------------------------------------------------

    # %%
    stored_indicators = {}
    time_dict = {}


    # 1. create indicators
    # ---------------------------------------------------------------------------------
    print('\n1. creating indicators...')
    start = time.time()
    start_indicator = datetime.now()
    if m in stored_indicators:
        indicators = stored_indicators[m]
    else:
        indicators = [dssp.int2indicator(A, m) for A in range(2**m)]
        indicators = np.asarray(indicators)
        stored_indicators[m]=indicators
    end = time.time()
    end_indicator = datetime.now()
    time_dict[seed] = dict()
    print_elapsed_time(end_indicator-start_indicator)
    time_dict[seed]['time_indicator'] = (end-start)/60
    # ---------------------------------------------------------------------------------


    # 2. calculate full set function v
    # ---------------------------------------------------------------------------------
    print('\n2. calculate full set function v...')
    start_v = datetime.now()
    start = time.time()
    gt_vecs = [np.asarray([v(x) for x in indicators])]
    end_v = datetime.now()
    end = time.time()
    print_elapsed_time(end_v-start_v)
    time_dict[seed]['time_full_set_function'] = (end - start) / 60
    # ---------------------------------------------------------------------------------


    # 3. calculate fourier transforms
    # ---------------------------------------------------------------------------------
    start = time.time()
    print('\n3. calculate Fourier transforms...')

    # A. WHT
    print('calculate WHT...')
    start_WHT = datetime.now()
    gt_whts = [dssp.fwht(vec.astype(np.float64)) for vec in gt_vecs][0]
    gt_whts = np.asarray(gt_whts)/2**m
    end_WHT = datetime.now()
    print_elapsed_time(end_WHT-start_WHT)
    #

    # B. FT3 = Polynomial Representation
    print('calculate FT3...')
    start_FT3 = datetime.now()
    gt_dsft3s = [dssp.fdsft3(vec.astype(np.float64)) for vec in gt_vecs][0]
    end_FT3 = datetime.now()
    print_elapsed_time(end_FT3-start_FT3)
    #

    # C. FT4
    print('calculate FT4...')
    start_FT4 = datetime.now()
    gt_dsft4s = [dssp.fdsft4(vec.astype(np.float64)) for vec in gt_vecs][0]
    end_FT4 = datetime.now()
    print_elapsed_time(end_FT4-start_FT4)
    #
    end = time.time()
    time_dict[seed]['time_fourier_transforms'] = (end - start) / 60
    # ---------------------------------------------------------------------------------




    # 4. save fourier transforms and timings
    # ---------------------------------------------------------------------------------
    print('\n4. saving Fourier transforms...')
    data_dict = {'m':m,
                 'seed':seed,
                 'vecs':gt_vecs,
                 'whts':gt_whts,
                 'dsft3s':gt_dsft3s,
                 'dsft4s':gt_dsft4s}
    pickle.dump(data_dict,open('spectral_analysis_precomputed_data.pkl', 'wb'))
    json.dump(time_dict, open('spectral_analysis_elapsed_times.json','w'))
    # ---------------------------------------------------------------------------------




    # 5. spectral plot
    # ---------------------------------------------------------------------------------
    print('\n5. plot spectral energy...')
    plot_spectrum(ft3 = gt_dsft3s,
                  ft4 = gt_dsft4s,
                  wht = gt_whts,
                  m=m,
                  targetpath=os.path.join('spectral_analysis_energy_distribution.pdf'),
                  flag_rescale=True,
                  )
    # ---------------------------------------------------------------------------------
