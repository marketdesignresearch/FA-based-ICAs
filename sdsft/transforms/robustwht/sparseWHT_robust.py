import numpy as np
from math import ceil, log, isclose, floor
from .hashing import Hashing

import dssp

def WHT(x):
    n = len(x.shape)
    s = x.reshape(-1)
    s_hat = ((1/2)**n)*dssp.fwht(s)
    x_hat = s_hat.reshape([2]*n)
    return x_hat
    

class SWHTRobust(object):
    def __init__(self, n, K, C=1.3, ratio=1.4):
        # C bucket constant
        self.C = C
        # Size of Ground set
        self.n = n
        # Sparsity
        self.K = K
        # Bucket halving ratio
        self.ratio = ratio

    def run(self, x):
        # B = no of bins we are hashing to
        # B = 48 * self.K
        B = int(self.K * self.C)
        # # print("B=", B)
        b = int(ceil(log(B, 2)))
        # T = no. of iterations
        # T = int(min(floor(log(B, 2)) - 1, ceil(log(self.K, 2)) + 1))
        # T = ceil(log(self.K,4))
        T = int(floor(log(B, self.ratio))) - 1
        # # print(T)
        # T = int(min(floor(log(B, 1.6)) - 1, 10*ceil(log(self.K, 2)) + 1))
        # # print(T)
        # current_estimate will hold as key frequencies and as value amplitudes
        current_estimate = {}
        for i in range(T):
            # # print("Iteration ", i)
            # # print("B=", B, "b=", b)
            # Define a new hashing matrix A
            hash = Hashing(self.n, b)
            # hashedEstimate will hold as keys bin frequencies and as values
            # tuples where the first element is the freq hashed to that bin
            # and second element is the amplitude
            hashed_current_estimate = self.hashFrequencies(hash, current_estimate)
            new_estimate = self.detectFrequency(x, hash, hashed_current_estimate)

            #########################
            # x.statistic(detectedFreq)
            # bucketCollision = {}
            # for edge in x.graph:
            #     freq = np.zeros((self.n))
            #     freq[edge[0]] = 1
            #     freq[edge[1]] = 1
            #     freq = tuple(freq)
            #     # print(edge, "hashed to ", hash.do_FreqHash(freq))
            #     try:
            #         bucketCollision[hash.do_FreqHash(freq)].append(edge)
            #     except KeyError:
            #         bucketCollision[hash.do_FreqHash(freq)] = []
            #         bucketCollision[hash.do_FreqHash(freq)].append(edge)
            # collisions = 0
            # for bucket in bucketCollision:
            #     if len(bucketCollision[bucket]) > 1:
            #         collisions += len(bucketCollision[bucket])
            #         # print(bucketCollision[bucket])
            # # print("collisions=", collisions)
            ##########################
            # Run iterative updates
            for freq in new_estimate:
                if freq in current_estimate:
                    current_estimate[freq] = current_estimate[freq] + new_estimate[freq]
                    if isclose(current_estimate[freq], 0.0, abs_tol=0.001):
                        # # print("deleting", freq)
                        del current_estimate[freq]

                else:
                    current_estimate[freq] = new_estimate[freq]

            # Buckets sizes for hashing reduces by half for next iteration
            B = int(ceil(B / self.ratio))
            b = int(ceil(log(B, 2)))
        return current_estimate

    def hashFrequencies(self, hash, est):
        # This function hashes the current estimated frequencies
        # of the signal to the buckets
        hashedEstimate = {}
        for key in est:
            hashed_key = hash.do_FreqHash(key)
            if hashed_key not in hashedEstimate:
                #  Initialize empty list
                hashedEstimate[hashed_key] = []
            hashedEstimate[hashed_key].append((key, est[key]))
        return hashedEstimate

    def detectFrequency(self, x, hash, hashed_current_estimate):

        # Detect Frequencies
        # Dictionary of detected frequencies
        detected_frequencies = {}
        detected_amplitudes = {}
        no_trials = 10
        # print("hashing matrix is:", hash.P)
        # print("hashed_current_estimate is:", hashed_current_estimate)
        for j in range(no_trials):
            # We need the WHT with a random shift for reference
            random_shift = np.random.randint(low=0, high=2, size=(self.n,))
            # print("randomshift", random_shift)
            hashed_signal = hash.do_TimeHashFunctional(x, random_shift)
            # # print("After Zero shift ", str(x.sampCplx))
            # # print("hashed_signal=", hashedSignal)
            ref_signal = WHT(hashed_signal)
            # print("reference signal")
            # print(ref_signal)
            # This dictionary will hold the WHTs of the subsampled signals
            hashedWHT = {}
            # Subsample Signal
            for j in range(self.n):
                # set a = e_j
                # # print("e=", j)
                e_j = np.zeros((self.n), dtype=np.int64)
                e_j[j] = 1
                # # print(a)
                hashed_signal = hash.do_TimeHashFunctional(x, (e_j + random_shift) % 2)
                hashedWHT[j] = WHT(hashed_signal)
                # print("e_", j, hashedWHT[j])

            # i is the number of the bucket we are checking in the iterations below
            for i in range(hash.B):
                bucket = self.toBinaryTuple(i, hash.b)
                # print("Bucket", bucket)
                # Compute the values of the current estimation of signal hashed to this bucket and subtract it off the
                # reference signal
                if bucket in hashed_current_estimate:
                    for X in hashed_current_estimate[bucket]:
                        if self._inp(X[0], random_shift) == 0:
                            ref_signal[bucket] = ref_signal[bucket] - X[1]
                        else:
                            ref_signal[bucket] = ref_signal[bucket] + X[1]

                # Only continue if a frequency with non-zero amplitude is hashed to bucket j
                # # print("cheching ref_signal", ref_signal[bucket])
                # print("ref_signal after subtraction", ref_signal)
                if isclose(ref_signal[bucket], 0.0, abs_tol=0.0001):
                    # # print("Entered if statement for ref_signal[bucket]=0")
                    continue
                # Subtract the values of the current estimation of signal hashed to this bucket and subtract it off the
                # signal with measurements
                for j in range(self.n):
                    e_j = np.zeros((self.n), dtype=np.int64)
                    e_j[j] = 1
                    if bucket in hashed_current_estimate:
                        for X in hashed_current_estimate[bucket]:
                            if self._inp(X[0], random_shift + e_j) == 0:
                                hashedWHT[j][bucket] = hashedWHT[j][bucket] - X[1]
                            else:
                                hashedWHT[j][bucket] = hashedWHT[j][bucket] + X[1]
                    # print("hashedWHT e_", j, "after subtraction", hashedWHT[j])
                # freq is the frequecy preset in this bin
                freq = [0] * self.n
                for j in range(self.n):
                    if np.sign(hashedWHT[j][bucket]) == np.sign(ref_signal[bucket]):
                        freq[j] = 0
                    else:
                        freq[j] = 1
                try:
                    detected_frequencies[tuple(freq)] += 1
                    if self._inp(freq, random_shift) == 0:
                        detected_amplitudes[tuple(freq)].append(ref_signal[bucket])
                    else:
                        detected_amplitudes[tuple(freq)].append(-ref_signal[bucket])
                except KeyError:
                    detected_frequencies[tuple(freq)] = 1
                    if self._inp(freq, random_shift) == 0:
                        detected_amplitudes[tuple(freq)] = [ref_signal[bucket]]
                    else:
                        detected_amplitudes[tuple(freq)] = [-ref_signal[bucket]]
                # print("detected_frequencies", detected_frequencies, "detected_amplitudes", detected_amplitudes)
            # # print (ref_signal)
        new_signal_estimate = {}
        # Take majority vote and median
        # print(detected_frequencies, "\n\n", detected_amplitudes, "\n\n")
        for freq in detected_frequencies:
            if detected_frequencies[freq] >= no_trials / 2:
                new_signal_estimate[freq] = np.median(detected_amplitudes[freq])

        return new_signal_estimate

    # This function computes the inner product of two 0-1 n-tuples
    @staticmethod
    def _inp(a, b):
        # # print("_inp", size(a), size(b))
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) % 2

    @staticmethod
    def toBinaryTuple(i, b):
        # Converts integer i into an b-tuple of 0,1s
        a = list(bin(i)[2:].zfill(b))
        a = tuple([int(x) for x in a])
        return a


