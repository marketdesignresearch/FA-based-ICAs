#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:06:20 2021

@author: chrisw
"""

import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import dssp as dssp

# %%
def accumulate(ft, n_groundset, metric=lambda x:np.linalg.norm(x), flag_rescale=False):
    popc = dssp.popcount(np.arange(2 ** n_groundset).astype(np.uint32))
    res = np.zeros(n_groundset + 1)
    for set_size in range(n_groundset + 1):
        if flag_rescale:
            res[set_size] = metric(ft[popc == set_size])/metric(ft)
        else:
            res[set_size] = metric(ft[popc == set_size])
    return np.arange(n_groundset+1), res

# %%
def confidence_plot(ax,
                    data,
                    label=None,
                    color='b',
                    do_confidence=False,
                    semilogy=False,
                    ylim=-1):
        if ylim == -1:
            if semilogy:
                ylim = (1e-8, 1)
            else:
                ylim = (0, 1)
        mean_nat = np.mean(data, axis=0)
        #x_axis = np.arange(len(mean_nat))
        x_axis = np.arange(mean_nat.shape[0])
        if semilogy:
            if ylim is not None:
                ylower = ylim[0]
            ax.semilogy(x_axis, mean_nat+ylower, '-o', label=label, color=color)
            ax.fill_between(x_axis, np.asarray([ylower]*len(mean_nat)), mean_nat+ylower, alpha=0.1, color=color)
        else:
            ax.plot(x_axis, mean_nat, '-o', label=label, color=color)
            ax.fill_between(x_axis, np.asarray([0]*len(mean_nat)), mean_nat, alpha=0.1, color=color)

        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, x_axis[-1])


# %%
def plot_spectrum(ft3,
                  ft4,
                  wht,
                  m,
                  flag_rescale=False,
                  metric=lambda x: np.linalg.norm(x)**2,
                  targetpath=None,
                  ):

    data_ft3 = []
    data_ft4 = []
    data_wht = []

    sizes, res_ft3 = accumulate(ft3, m, metric, flag_rescale=flag_rescale)
    data_ft3 += [res_ft3]
    sizes, res_ft4 = accumulate(ft4, m, metric, flag_rescale=flag_rescale)
    data_ft4 += [res_ft4]
    sizes, res_wht = accumulate(wht, m, metric, flag_rescale=flag_rescale)
    data_wht += [res_wht]

    data_ft3 = np.asarray(data_ft3)
    data_ft4 = np.asarray(data_ft4)
    data_wht = np.asarray(data_wht)

    n_rows = 1
    n_cols = 3
    label = 'Spectral Energy of v(x)'
    label_fontsize = 25

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(27, 9), facecolor='w', edgecolor='k')

    confidence_plot(ax[0], data_ft3, label=label, color='r')
    ax[0].set_title('Polynomial Representation',fontsize=label_fontsize)
    ax[0].set_xticks(np.arange(0, m+1, 1))
    ax[0].set_ylabel('Spectral Energy in %',fontsize=label_fontsize)
    ax[0].set_xlabel('Bundle (i.e., Set) Size',fontsize=label_fontsize)
    ax[0].legend(loc=1,fontsize=label_fontsize-5)
    ax[0].grid()


    confidence_plot(ax[1], data_ft4, label=label, color='b')
    ax[1].set_title('FT4',fontsize=label_fontsize)
    ax[1].set_xticks(np.arange(0, m+1, 1))
    ax[1].set_xlabel('Bundle (i.e., Set) Size',fontsize=label_fontsize)
    ax[1].legend(loc=1,fontsize=label_fontsize-5)
    ax[1].grid()

    confidence_plot(ax[2], data_wht, label=label, color='g')
    ax[2].set_title('WHT',fontsize=label_fontsize)
    ax[2].set_xticks(np.arange(0, m+1, 1))
    ax[2].set_xlabel('Bundle (i.e., Set) Size',fontsize=label_fontsize)
    ax[2].legend(loc=1,fontsize=label_fontsize-5)
    ax[2].grid()

    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[2].get_yticklabels(), visible=False)

    if targetpath is not None:
        plt.savefig(targetpath, format='pdf')
    plt.tight_layout()
    plt.show()

# %%
def timediff_d_h_m_s(td):

    """Measures time difference in days, hours,minutes and seconds.

    Arguments
    ----------
    td :
        A timedelta object from the datetime package, representing the difference between two
        datetime.times.

    Return
    ----------
    A tuple representing the td object as days, hours, minutes, and seconds.

    """

    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return (
            -(td.days),
            -int(td.seconds / 3600),
            -int(td.seconds / 60) % 60,
            -(td.seconds % 60),
        )
    return (
        td.days,
        int(td.seconds / 3600),
        int(td.seconds / 60) % 60,
        td.seconds % 60,
    )

# %%
def print_elapsed_time(diff):
    print("elapsed time: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(diff)),
                        "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",)

