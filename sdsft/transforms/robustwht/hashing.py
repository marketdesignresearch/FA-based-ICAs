import numpy as np
from numpy import matrix
from numpy import linalg
import math


class Hashing(object):
    def __init__(self, n, b):
        # Permutation Matrix is an n * b array
        self.P = np.random.randint(low=0, high=2, size=(n, b))
        ##print("P is", self.P.transpose())
        # n is dimensionality of signal
        self.n = n
        # B = 2^b = no. of buckets we are hashing into
        self.b = b
        self.B = 2 ** self.b
        self.indicators = np.asarray([self.__to_binary(A) for A in range(self.B)],dtype=np.int32)
        self.cache = {}
        
        
    def do_TimeHashFunctional(self, signal, shift):
        shifted_indices = np.dot(self.indicators, self.P.T) + shift[np.newaxis]
        shifted_indices = np.mod(shifted_indices, 2)
        out = signal(shifted_indices)
        return out.reshape([2]*self.b)

    def do_TimeHash(self, signal, shift):
        key = tuple(np.squeeze(shift).tolist())
        try:
            return self.cache[key]
        except KeyError:
            pass

        assert (signal.shape == tuple([2] * self.n)
                ), "Signal shape does not match hashing matrix dimension"
        # assert(shift.shape == (self.n, 1)), "Shift shape does not match hashing matrix dimension"
        out = np.zeros(shape=tuple([2] * self.b), dtype=np.float64)
        for i in range(self.B):
            t = self.__to_binary(i)
            # print("i = ", i)
            # print("t = ", t)
            # print(shift)
            index = (np.dot(self.P, t) + shift) % 2
            # print(t, index)
            # print("index = ", index)
            out[tuple(np.squeeze(t))] = signal[tuple(np.squeeze(index))]
        self.cache[key] = out
        return out

    def do_FreqHash(self, freq):
        # print(self.P)
        f = np.array(freq, dtype=np.intc)
        hashed_f = np.dot(np.transpose(self.P), f) % 2
        return tuple(hashed_f.tolist())

    def __to_binary(self, i):
        # Converts integer i into an (n,1) 0-1 vector
        a = list(bin(i)[2:].zfill(self.b))
        a = [int(x) for x in a]
        a = np.array(a, dtype=np.intc)
        return a


class InvertibleHashing(Hashing):

    def __init__(self, n, b):
        self.n = n
        # B = 2^b = no. of buckets we are hashing into
        self.b = b
        self.B = 2 ** self.b
        # Hashing matrix is an n \times b matrix
        self.P = np.random.randint(low=0, high=2, size=(n, b))
        # print("P is", self.P)
        self.rank_P, _ = InvertibleHashing.__reduced_row_echelon(self.P)
        # print("rank of P is", self.rank_P, "and its reduced rowEchelon form is ", _)
        self.extended_P = self.P
        current_rank, _ = InvertibleHashing.__reduced_row_echelon(self.extended_P)
        for r in range(self.n - self.rank_P):
            new_column = np.random.randint(low=0, high=2, size=(n, 1))
            # print("adding column", new_column)
            temp_extended_P = np.concatenate((self.extended_P, new_column), axis=1)
            # print("temp extended p=", temp_extended_P)
            new_rank, _ = InvertibleHashing.__reduced_row_echelon(temp_extended_P)
            # print("reduced row for of extended p is", _)
            # print("new rank=", new_rank)
            while new_rank == current_rank:
                new_column = np.random.randint(low=0, high=2, size=(n, 1))
                # print("New column didnt increase rnak so trying another column", new_column)
                temp_extended_P = np.concatenate((self.extended_P, new_column), axis=1)
                new_rank, _ = InvertibleHashing.__reduced_row_echelon(temp_extended_P)
            current_rank += 1
            self.extended_P = np.concatenate((self.extended_P, new_column), axis=1)
            # print(self.extended_P)

        self.measurement_matrix = self.extended_P[:, self.b:].transpose()
        self.no_binary_measurements = self.n - self.rank_P
        self.whole_hashing_matrix = np.transpose(self.extended_P)
        self.cache = {}

    def inverse_freq_hash(self, freq):
        freq = np.array(freq, dtype=np.intc)
        freq = np.expand_dims(freq, axis=1)
        # print(freq, freq.shape)
        linear_system = np.concatenate((self.whole_hashing_matrix, freq), axis=1)
        _, reduced_row_form = InvertibleHashing.__reduced_row_echelon(linear_system)
        # print(reduced_row_form)
        inverse_freq = reduced_row_form[0:self.n, self.n]
        # return np.sum((np.dot(self.whole_hashing_matrix, inverse_freq)+ np.squeeze(freq))%2)==0
        return tuple(inverse_freq)

    @staticmethod
    def __reduced_row_echelon(matrix):
        # This functions returns False and an empty matrix if the matrix is not invertible and True and the inverse if it is
        # We use gaussian elimination
        matrix = np.copy(matrix)
        m, n = matrix.shape
        rank = 0
        for col in range(n):
            # This for loop finds a row with a leading one
            # print("col", col)
            for row in range(rank, m):
                if matrix[row, col] == 1:
                    break
            else:
                continue
            # print("swapping", row, rank)
            # Swap two rows
            matrix[[row, rank], :] = matrix[[rank, row], :]
            # print(matrix)
            for row in range(m):
                if matrix[row, col] == 1 and row != rank:
                    matrix[row, :] = (matrix[row, :] + matrix[rank, :]) % 2
            # print("after subtracction", matrix)
            rank += 1
        return rank, matrix
if __name__ == "__main__":
    #np.random.seed(10)
    x = np.random.randint(0, high=5, size=(2, 2, 2))
    print(x)
    print(x[(0, 0, 0)])
    print(x[(0, 0, 1)])
    print(x[(0, 1, 0)])
    print(x[(0, 1, 1)])
    print(x[(1, 0, 0)])
    print(x[(1, 0, 1)])
    print(x[(1, 1, 0)])
    print(x[(1, 1, 1)])

    h = Hashing(n=3, b=2)
    shift = np.array([[0], [0], [0]], dtype=np.intc)
    # out = h.do_TimeHash(signal = x, shift = shift)
    # print ("Output is", out)
    x = np.random.random(1024 ** 2)
    print(x.shape)
    h = Hashing(n=4, b=2)
    # (h.__to_binary(642))
    print(h.do_FreqHash((0, 1, 0, 0)))
    h = InvertibleHashing(20, 4)
    print(h.extended_P.shape)
    success = 0
    for _ in range(10000):
        measurment = np.random.randint(low=0, high=2, size=20)
        if h.inverse_freq_hash(measurment):
            success += 1
    print(success)
