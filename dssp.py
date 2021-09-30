import numpy as np
from tqdm import tqdm
import time


# Items [1,...,M] x [1,...,M] builds the index tensor
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    # shape = [len(a) for a in arrays]
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr  # arr.reshape(tuple(shape))


# subsetting DSFT for shift 2 to specific rows and columns defined by (rows, columns), i.e. specific bundles
def submatrix_F2(rows, columns):
    prod = cartesian_product(rows, columns)
    return ((prod[:, :, 0] & prod[:, :, 1]) == prod[:, :, 0])

# ----------------------------------------------------------------------------------------------------------------------------------------
# subsetting inverse DSFTs for shift 2,3,4 and WHT to specific rows and columns defined by (rows, columns), i.e. specific bundles


def submatrix_iF2(rows, columns):
    prod = cartesian_product(rows, columns)
    iota = ((prod[:, :, 0] & prod[:, :, 1]) == prod[:, :, 0])
    intersections = (prod[:, :, 1] & (~prod[:, :, 0]))
    cards = []
    for row in intersections:
        cards.append([pypopcount(entry) for entry in row])
    cards = np.asarray(cards)
    return ((-1)**(cards)) * iota


def submatrix_WHT(rows, columns):
    prod = cartesian_product(rows, columns)
    intersections = prod[:, :, 0] & prod[:, :, 1]
    cards = []
    for row in intersections:
        cards.append([pypopcount(entry) for entry in row])
    cards = np.asarray(cards)
    return ((-1)**(cards)).astype(np.float64)

def submatrix_iF3(rows, columns):
    prod = cartesian_product(rows, columns)
    idsft3 = submatrix_WHT(rows, columns)
    idsft3[prod[:, :, 0] & prod[:, :, 1] != prod[:, :, 1]] = 0
    return idsft3.astype(np.float64)

def submatrix_iF4(rows, columns):
    prod = cartesian_product(rows, columns)
    return (prod[:, :, 0] & prod[:, :, 1]) == 0



# ----------------------------------------------------------------------------------------------------------------------------------------


def reconstruct4(IF_MP, s_M, support, n):
    sparse = np.zeros(2**n)
    try:
        hats_B = np.linalg.solve(IF_MP, s_M)
    except np.linalg.LinAlgError:
        print('Linalgerror: Sampling GLS not of full rank, using leastsquares...')
        hats_B, _, _, _ = np.linalg.lstsq(IF_MP, s_M)
    sparse[support] = hats_B
    return fidsft4(sparse), hats_B


def eval_dsft4_bandlimited(A, fourier_coefs, fourier_support, N):
    A_c = 2**N - 1 - A
    return np.sum(fourier_coefs[(fourier_support & A_c) == fourier_support])


def eval_dsft3_bandlimited(A, fourier_coefs, fourier_support, N):
    mask = (fourier_support & A) == fourier_support
    cardinalities = np.asarray([pypopcount(B) for B in fourier_support[mask]])
    return np.sum(fourier_coefs[mask] * (-1)**cardinalities)


def conv3(h, s):
    frequency_response = fr(h)
    s_hat = fdsft3(s)
    return fidsft3(frequency_response * s_hat)


def conv4(h, s):
    frequency_response = fr(h)
    s_hat = fdsft4(s)
    return fidsft4(frequency_response * s_hat)


def fr(signal):
    return fidsft4(signal)


# Fast discrete set fourier transforms
def fdsft3(signal):
    N = len(signal)
    h = 1
    transform = signal.copy()
    while h < N:
        for i in range(0, N, 2*h):
            for j in range(i, i+h):
                x = transform[j]
                y = transform[j + h]
                transform[j] = x
                transform[j + h] = x - y
        h *= 2
    return transform


def fidsft3(transform):
    return fdsft3(transform)


def fwht(signal):
    N = len(signal)
    h = 1
    transform = signal.copy()
    while h < N:
        for i in range(0, N, 2*h):
            for j in range(i, i+h):
                x = transform[j]
                y = transform[j + h]
                transform[j] = x + y
                transform[j + h] = x - y
        h *= 2
    return transform


def fdsft4(signal):
    N = len(signal)
    h = 1
    transform = signal.copy()
    while h < N:
        for i in range(0, N, 2*h):
            for j in range(i, i+h):
                x = transform[j]
                y = transform[j + h]
                transform[j] = y
                transform[j + h] = x - y
        h *= 2
    return transform


def fidsft4(signal):
    N = len(signal)
    h = 1
    transform = signal.copy()
    while h < N:
        for i in range(0, N, 2*h):
            for j in range(i, i+h):
                x = transform[j]
                y = transform[j + h]
                transform[j] = x + y
                transform[j + h] = x
        h *= 2
    return transform


# counts items in bundles ordered lexicographically 1:x_1, 2:x_2,3:{x_1,x_2}
def popcount(arr):
    out = np.asarray([pypopcount(A) for A in arr])
    return out


def pypopcount(n):
    """ this is actually faster """
    return bin(n).count('1')


# Transform integer representing set A to indicator vector representing A. Enumeration done lexycographically. 0 -> \empty, 1 -> [1,0,...], 2 -> [0,1,0,...], 3->[1,1,0,...], , 4->[0,0,1,...]
def int2indicator(A, n_groundset, reverse=False):
    indicator = [int(b) for b in bin(2**n_groundset + A)[3:][::-1]]
    indicator = np.asarray(indicator, dtype=np.bool)
    if reverse:
        indicator = np.flip(indicator, axis=0)
    return indicator


def indicator2int(indicator, n_groundset, reverse=False):
    singletons = 2**np.arange(n_groundset)
    if reverse:
        singletons = np.flip(singletons, axis=0)
    return np.sum(singletons[indicator.astype(np.bool)])


def int2elements(A, n_groundset, reverse=False):
    indicator = int2indicator(A, n_groundset)
    singletons = 2**np.arange(n_groundset)
    if reverse:
        singletons = np.flip(singletons, axis=0)
    return singletons[indicator]


# Generate all bids for a single Auction instance: world and all bidders
# import itertools
def generate_all_bids(world):
    N = world.get_bidder_ids()
    M = world.get_good_ids()
    print()
    # bundle_space = list(itertools.product([0, 1], repeat=len(M)))
    bundle_space = [[int(b) for b in bin(2**len(M) + k)[3:][::-1]] for k in np.arange(2**len(M))]
    s = time.time()
    bundle_value_pairs = np.array([list(x)+[world.calculate_value(bidder_id, x) for bidder_id in N] for x in tqdm(bundle_space)])
    e = time.time()
    print('Elapsed sec: ', round(e-s))
    return(bundle_value_pairs)


def shift(Q, s, model=3):
    """
    :param Q: integer representation of subset Q
    :param s:
    :param model:
    :return:
    """
    N = int(np.log2(len(s)))
    subsets = np.arange(2**N, dtype=np.int64)
    if model == 1:
        if pypopcount(Q) != 1:
            raise NotImplementedError('model 1 is not implemented')
        shifted = s + shift(Q, s, 3)
        shifted[subsets & Q != Q] = 0
        return shifted
    elif model == 2:
        if pypopcount(Q) != 1:
            raise NotImplementedError('model 1 is not implemented')
        shifted = s + shift(Q, s, 4)
        shifted[subsets & Q == Q] = 0
        return shifted
    elif model == 3:
        return s[subsets & (~Q)]
    elif model == 4:
        return s[subsets | Q]
    elif model == 5:
        return s[(subsets & (~Q)) | (Q & (~subsets))]
    else:
        raise NotImplementedError('model not implemented')

# counterpart to <<satsinstance>>.generate_random_bids
# here !truly uniform! sampling over 2^m bundles where m is # of items
# value_model...instance of lsvm
# size...number of desired uniform bids
def UNIF_random_bids(value_model, bidder_id, size):
    ncol = len(value_model.get_good_ids())  # number of items in value model
    D = np.unique(np.random.choice(2, size=(size, ncol)), axis=0)
    # get unique ones if accidently sampled equal bundle
    while D.shape[0] != size:
        tmp = np.random.choice(2, size=ncol).reshape(1, -1)
        D = np.unique(np.vstack((D, tmp)), axis=0)
    # define helper function for specific bidder_id

    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return(D)
