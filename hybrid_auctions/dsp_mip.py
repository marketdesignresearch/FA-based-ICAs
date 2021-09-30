# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file implmements the class DspMip (Discrete Signal Processing-based Mixed Integer Program). This class is used for solving the dsp-based Winner Determination Problems for shifts: 'shift5' (= WHT) and 'shift4'.



NN_MIP has the following functionalities:
    0.CONSTRUCTOR: __init__(self, models, =None)
        models = a dict of dicts with (key,value) pairs defined as: ('Bidder_i', {'v_hat': fourier coefficients, 'W': bit-matrix, 'dsp_type': string denoting the shift used. Either 'shift5' or 'shift4'})
    1.METHOD: __repr__(self)
        Echoe on on your python shell when it evaluates an instances of this class
    2.METHOD: print_optimal_allocation(self)
        Prints the optimal allocation.
    3.METHOD: print_mip_constraints(self)
        Prints the MIP constraints of the DNN-based WDP.
    4.METHOD: print_solution_values(self)
        Prints solution values of the mip variables.
    5.METHOD: initialize_mip(self, M=None, feasibility_equality=False):
        M = big-M variable which is set gloabally for each constraint in the WDP.
        feasibility_equality = boolean, if each item has to be sold.
        This method initializes the MIP forof the dsp based WDP.
    6.METHOD: solve_mip(self, log_output=False, time_limit=None, mip_relative_gap=None, mip_start=None, integrality_tol=None)
        log_output = boolean for detailed output of cplex when solving the MIP
        time_limit = time limit when solving the MIP
        mip_relative_gap = relative gap limit when solving the MIP
        mip_start = a SolveSolution instance imported from docplex.mp.solution used for a warm start when solving the MIP
        integrality_tol = tolerance for when a number is intepreted as integer.
        This method returns a docplex.mp.solution.SolveSolution() instance.
    7.METHOD: reset_mip(self):
         This method resets the MIP

"""

# Libs
import numpy as np
import pandas as pd
import docplex.mp.model as cpx
import re
import logging
# %% DSP MIP CLASS


class DspMip:

    def __init__(self, models):
        self.models = models  # models represented as tuple (v_hat, W, dsp_type)
        self.bidders = list(models.keys())
        self.bidders.sort()
        self._check_dsp_types()  # check if dsp types are admisible
        self.n = len(models)  # number of bidders
        self._check_item_dimension() # check if all bidders have the same item dimension
        self.m = models[list(models.keys())[0]]['W'].shape[1]  # number of items
        self.bidder_sparsities = {bidder_key: self.models[bidder_key]['W'].shape[0] for bidder_key in self.bidders}  # k_i's per bidder
        self._check_v_hat_dimension()
        self.soltime = None  # timing
        self.optimal_allocation = {bidder_key: np.ones(self.m, dtype=int)*(-1) for bidder_key in self.bidders}  # init optimal allocation


        # init MIP parameters
        self.mip = cpx.Model(name="dsp_based_MIP")  # MIP instance
        self.M = None  # big-M constraint
        self.objectives = {bidder_key: None for bidder_key in self.bidders}  # objectives for individual bidders
        self.a = {}  # MIP variable a
        self.z = {}  # MIP variable z
        self.s = {}  # MIP variable s (only for shift5)
        self.y = {}  # MIP variable y

    def print_optimal_allocation(self):
        D = pd.DataFrame.from_dict(self.optimal_allocation, orient='index')
        D.columns = ['Item_{}'.format(j) for j in range(1, self.m+1)]
        D.loc['Sum'] = D.sum(axis=0)
        pd.set_option('display.max_columns', self.m)
        print(D)
        del D


    def print_mip_constraints(self):
        print('############################### CONSTRAINTS ###############################')
        print()
        k = 0
        for m in range(0, self.mip.number_of_constraints):
            if self.mip.get_constraint_by_index(m) is not None:
                print('({}):   '.format(k), self.mip.get_constraint_by_index(m))
                print()
                k = k+1
            if self.mip.get_indicator_by_index(m) is not None:
                print('({}):   '.format(k), self.mip.get_indicator_by_index(m))
                print()
                k = k+1
        print('\n')


    def print_solution_values(self):
        if self.mip.number_of_variables==0:
            print('No variables initialized!')
            return(None)
        if self.mip.number_of_variables>0 and self.mip.get_solve_details() is None:
            print('Mip not yet solved!')
        else:
            print()
            print('Optimal values of all Variables:')
            print()
            for m in range(self.mip.number_of_variables):
                var = self.mip.get_var_by_index(m)
                tab_space = len(str(m))
                print('{}:'.format(m) + ''.join((8-tab_space)*[' ']) + '{} = {}'.format(var,var.solution_value))


    def initialize_mip(self, M=None, feasibility_equality=False, bidder_specific_constraints=None, GSVM_specific_constraints=False, national_circle_complement=None):
        # set big-M constraint
        self.M = M
        # shift dependent constraints for bidder i
        for bidder_key in self.bidders:
             self._add_constraints(bidder_key=bidder_key)
             self._add_objective(bidder_key=bidder_key)
         # allocation constraints for a^i's
        for j in range(self.m):
             if feasibility_equality:
                 self.mip.add_constraint(ct=(self.mip.sum(self.a[(self._key_to_int(bidder_key), j)] for bidder_key in self.bidders) == 1),
                                         ctname="Feasability_CT_item{}".format(j))
             else:
                 self.mip.add_constraint(ct=(self.mip.sum(self.a[(self._key_to_int(bidder_key), j)] for bidder_key in self.bidders) <= 1),
                                         ctname="Feasability_CT_item{}".format(j))
        # bidder specific constraints
        if bidder_specific_constraints is not None:
            self._add_bidder_specific_constraints(bidder_specific_constraints=bidder_specific_constraints)

        # GSVM specific allocation constraints for regional and local bidder
        if GSVM_specific_constraints and national_circle_complement is not None:
            for bidder_key in self.bidders:
                # regional bidder
                if bidder_key in ['Bidder_0','Bidder_1','Bidder_2','Bidder_3','Bidder_4','Bidder_5']:
                    logging.debug('Adding GSVM specific constraints for regional {}.'.format(bidder_key))
                    self.mip.add_constraint(ct=(self.mip.sum(self.a[(self._key_to_int(bidder_key), j)] for j in range(0, self.m)) <= 4), ctname="GSVM_CT_Regional{}".format(bidder_key))
                # national bidder
                elif bidder_key in ['Bidder_6']:
                    logging.debug('Adding GSVM specific constraints for national {} with national circle complement {}.'.format(bidder_key,national_circle_complement))
                    self.mip.add_constraint(ct=(self.mip.sum(self.a[(self._key_to_int(bidder_key), j)] for j in national_circle_complement) == 0), ctname="GSVM_CT_National{}".format(bidder_key))
                else:
                    raise NotImplementedError('GSVM only implmented in default version for Regional Bidders:[Bidder_0,..,Bidder_5] and National Bidder: [Bidder_6]. You entered {}'.format(bidder_key))

        # add objective
        objective = self.mip.sum(self.objectives)
        self.mip.maximize(objective)


    def solve_mip(self, log_output=False, time_limit=None, mip_relative_gap=None, mip_start=None, integrality_tol=None):
        if mip_start is not None:
            self.mip
            self.mip.add_mip_start(mip_start)
        if time_limit is not None:
            self.mip.set_time_limit(time_limit)
        if mip_relative_gap is not None:
            self.mip.parameters.mip.tolerances.mipgap = mip_relative_gap
        if integrality_tol is not None:
            self.mip.parameters.mip.tolerances.integrality.set(integrality_tol)
        logging.debug('Time Limit of %s', self.mip.get_time_limit())
        logging.debug('Mip relative gap %s', self.mip.parameters.mip.tolerances.mipgap.get())
        Sol = self.mip.solve(log_output=log_output)
        self.soltime = Sol.solve_details._time
        logging.debug(self.mip.get_solve_status())
        logging.debug(self.mip.get_solve_details())
        logging.debug('Objective Value: %s \n',self.mip.objective_value)
        # set solution
        for bidder_key in self.bidders:
            i = self._key_to_int(bidder_key)
            self.optimal_allocation[bidder_key] = np.asarray([int(self.a[(i, j)].solution_value) for j in range(self.m)])
        return(Sol)


    def reset_mip(self):
        self.mip = cpx.Model(name="MIP")


    def _key_to_int(self, key):
        return(int(re.findall(r'\d+', key)[0]))


    def _check_dsp_types(self):
        admissible_dsp_types = ['shift5', 'shift4', 'shift3']
        input_dsp_types = [self.models[bidder_key]['dsp_type'] for bidder_key in self.bidders]
        if not all([k in admissible_dsp_types for k in input_dsp_types]):
            raise ValueError('At least one of the input dsp_types: {} is not part of the admissible types: '.format(input_dsp_types) + str(admissible_dsp_types))


    def _check_item_dimension(self):
        item_dimensions = [self.models[bidder_key]['W'].shape[1] for bidder_key in self.bidders]
        if len(set(item_dimensions)) > 1:
            raise ValueError('Item dimensions, i.e., W.shape[0]: {}  differ for some bidders!'.format(item_dimensions))


    def _check_v_hat_dimension(self):
     if not all([self.models[bidder_key]['v_hat'].shape == (self.models[bidder_key]['W'].shape[0], ) for bidder_key in self.bidders]):
            raise ValueError('Row dimensions of Ws, i.e., sparsities: {} differ from dimensions of v_hats: {} (expected column vector of shape (k,), where k is the sparsity) !'.format(self.bidder_sparsities,
                            {bidder_key: self.models[bidder_key]['v_hat'].shape for bidder_key in self.bidders}))


    def _add_constraints(self, bidder_key):
        i = self._key_to_int(bidder_key)  # integer key of bidder i
        W = self.models[bidder_key]['W']  # W matrix of bidder i
        K_i = self.bidder_sparsities[bidder_key]  # Sparsity of bidder i, i.e., number of rows of W

        # WHT = shift5 constraints
        if self.models[bidder_key]['dsp_type'] == 'shift5':
            # initialize MIP variables
            self.a.update({(i,j): self.mip.binary_var(name="a({}{})".format(i, j)) for j in range(self.m)})  # binary variables for allocation a_i
            self.s.update({(i,k): self.mip.integer_var(lb=0, name="s({}{})".format(i, k)) for k in range(K_i)})  # slack variable s_i
            self.y.update({(i,k): self.mip.binary_var(name="y({}{})".format(i,k)) for k in range(K_i)})  # binary variables y_i
            # add linear matrix constraint: Wa_i-2s_i = y_i
            for k in range(K_i):
                self.mip.add_constraint(ct=(self.mip.sum(W[k, j]*self.a[(i, j)] for j in range(self.m)) - 2*self.s[(i, k)] == self.y[(i,k)]), ctname="shift5_CT_Bidder{}_Row{}".format(i, k))

        # shift4 constraints
        if self.models[bidder_key]['dsp_type'] == 'shift4':
            # initialize MIP variables
            self.a.update({(i,j): self.mip.binary_var(name="a({}{})".format(i, j)) for j in range(self.m)})  # binary variables for allocation a_i
            self.z.update({(i,k): self.mip.continuous_var(lb=0, name="z({}{})".format(i, k)) for k in range(K_i)})  # slack variable z_i
            self.y.update({(i,k): self.mip.binary_var(name="y({}{})".format(i,k)) for k in range(K_i)})  # binary variables y_i
            # add linear matrix constraint
            for k in range(K_i):
                self.mip.add_constraint(ct=(self.z[(i, k)] >= 1-self.mip.sum(W[k, j]*self.a[(i, j)] for j in range(self.m))), ctname="shift4_CT1_Bidder{}_Row{}".format(i, k))
                self.mip.add_constraint(ct=(self.z[(i, k)] <= (1-self.mip.sum(W[k, j]*self.a[(i, j)] for j in range(self.m))) + self.M*self.y[(i,k)]), ctname="shift4_CT2_Bidder{}_Row{}".format(i, k))
                self.mip.add_constraint(ct=(self.z[(i, k)] <= self.M*(1-self.y[(i,k)])), ctname="shift4_CT3_Bidder{}_Row{}".format(i, k))

        # shift3 constraints
        if self.models[bidder_key]['dsp_type'] == 'shift3':
            # initialize MIP variables
            self.a.update({(i,j): self.mip.binary_var(name="a({}{})".format(i, j)) for j in range(self.m)})  # binary variables for allocation a_i
            self.z.update({(i,k): self.mip.continuous_var(lb=0, name="z({}{})".format(i, k)) for k in range(K_i)})  # slack variable z_i
            self.y.update({(i,k): self.mip.binary_var(name="y({}{})".format(i,k)) for k in range(K_i)})  # binary variables y_i
            # add linear matrix constraint
            for k in range(K_i):
                self.mip.add_constraint(ct=(self.z[(i, k)] >= 1-self.mip.sum(W[k, j]*(1-self.a[(i, j)]) for j in range(self.m))), ctname="shift3_CT1_Bidder{}_Row{}".format(i, k))
                self.mip.add_constraint(ct=(self.z[(i, k)] <= (1-self.mip.sum(W[k, j]*(1-self.a[(i, j)]) for j in range(self.m))) + self.M*self.y[(i,k)]), ctname="shift3_CT2_Bidder{}_Row{}".format(i, k))
                self.mip.add_constraint(ct=(self.z[(i, k)] <= self.M*(1-self.y[(i,k)])), ctname="shift3_CT3_Bidder{}_Row{}".format(i, k))

    #TODO: implement Gianlucas integer cut
    def _add_bidder_specific_constraints(self, bidder_specific_constraints):
        for bidder_key, bundles in bidder_specific_constraints.items():
                bidder_id = np.where([x==bidder_key for x in self.bidders])[0][0]
                count=0
                logging.debug('Adding bidder specific constraints')
                for bundle in bundles:
                    #logging.debug(bundle)
                    self.mip.add_constraint(ct=(self.mip.sum((self.a[(bidder_id, j)]==bundle[j]) for j in range(0, self.m))<=(self.m-1)),
                                                ctname="BidderSpecificCT_Bidder{}_No{}".format(bidder_id,count))
                    count= count+1

    def _add_objective(self, bidder_key):
        i = self._key_to_int(bidder_key)  # integer key bidder i
        v_hat = self.models[bidder_key]['v_hat']  # estimated Fourier coefficients: v_hat of bidder i
        K_i = self.bidder_sparsities[bidder_key]  # Sparsity of bidder i, i.e., number of rows of W

        # WHT = shift5 objective
        if self.models[bidder_key]['dsp_type'] == 'shift5':
            self.objectives[bidder_key] = self.mip.sum(v_hat[k]*(-2*self.y[(i,k)]+1) for k in range(K_i))

        # shift4 objective
        if self.models[bidder_key]['dsp_type'] == 'shift4':
            self.objectives[bidder_key] = self.mip.sum(v_hat[k]*self.z[(i,k)] for k in range(K_i))

        # shift3 objective
        if self.models[bidder_key]['dsp_type'] == 'shift3':
            self.objectives[bidder_key] = self.mip.sum(v_hat[k]*self.z[(i,k)] for k in range(K_i))