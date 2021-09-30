# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:54 2020

"""

import pickle
import numpy as np
from .common import SetFunction, SparseDSFT4Function, SparseWHTFunction

class Bidder(SetFunction):
    def __init__(self, world, bidder_id):
        self.world = world
        self.bidder_id = bidder_id
        self.call_counter = 0

    def __call__(self, indicators, count_flag=True):
        if not isinstance(indicators, np.ndarray):
            inds = indicators.toarray()
        else:
            inds = indicators
        if len(inds.shape) < 2:
            inds = inds[np.newaxis, :]
        if count_flag:
            self.call_counter += inds.shape[0]
        values = [self.world.calculate_value(self.bidder_id, ind) for ind in inds]
        return np.asarray(values)

def bidders2dict(estimates, model):
    bidders = {}
    for i, estimate in enumerate(estimates):
        bidders['Bidder_%d'%i] = {'v_hat':estimate.coefs.flatten(),
               'W':estimate.freqs.astype(np.int32),
               'dsp_type': 'shift%d'%model}
    return bidders


def serializeBidders(fname, estimates, model):
    bidders = bidders2dict(estimates, model)
    with open(fname, 'wb') as handle:
        pickle.dump(bidders, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadBidders(fname):
    bidders=[]
    with open(fname, 'rb') as handle:
        b = pickle.load(handle)
    for bidder_id in range(len(list(b.keys()))):
        freqs = b['Bidder_%d'%bidder_id]['W']
        coefs = b['Bidder_%d'%bidder_id]['v_hat']
        sf_type =  b['Bidder_%d'%bidder_id]['dsp_type']
        if sf_type == 'shift4':
            bidder = SparseDSFT4Function(freqs, coefs)
        elif sf_type== 'shift5':
            bidder = SparseWHTFunction(freqs, coefs)
        bidders += [bidder]
    return bidders

def loadBiddersDict(fname):
    with open(fname, 'rb') as handle:
        b = pickle.load(handle)
    for bidder in b.keys():
        if b[bidder]['dsp_type'] == 'Shift 5':
            b[bidder]['dsp_type'] = 'shift5'
        elif b[bidder]['dsp_type'] == 'Shift 4':
            b[bidder]['dsp_type'] = 'shift4'
    return b

