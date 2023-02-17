# Fourier Analysis-based Iterative Combinatorial Auctions

Published at [IJCAI 2022](https://ijcai-22.org/)

This piece of software provides tools for **set function Fourier analysis**. Details can be found in the following paper:

**[Fourier Analysis-based Iterative Combinatorial Auctions](https://arxiv.org/abs/2009.10749)**<br/>
Jakob Weissteiner, Chris Wendler, Sven Seuken, Ben Lubin, and Markus Püschel.<br/>
*In Proceedings of the Thirty-first International joint Conference on Artificial Intelligence IJCAI'22, Vienna, AUT, July 2022.*<br/>
Full paper version including appendix: [Fourier Analysis-based Iterative Combinatorial Auctions](https://arxiv.org/abs/2009.10749)

If you use this software for academic purposes, please cite the above in your work. Bibtex for this reference is as follows:

```tex
@InProceedings{weissteiner2022fourier,
  author    = {Weissteiner, Jakob and Wendler, Chris and Seuken, Sven and Lubin, Ben and P{\"u}schel, Markus},
  title     = {Fourier Analysis-based Iterative Combinatorial Auctions},
  booktitle = {Proceedings of the 31st International Joint Conference on Artificial Intelligence},
  year      = {2022},
}
```

Specifically, this piece of software enables

**i. in Section 3.1**

to analyse the ***spectral energy distribution*** in three different Fourier domains: FT3, FT4, and WTH of your own set function $v:\\\{0,1\\\}^m\to \mathbb{R}$ (note that $v(\cdot)$ does not necessarily has to be a bidder's value function but can be an arbitrary set function).

**ii. in Section 3.2**

to ***apply set function Fourier analysis to combinatorial auctions*** and compute the outcome of the HYBRID mechanisms and MLCA, which are described in detail in the paper [Fourier Analysis-based Iterative Combinatorial Auctions](https://arxiv.org/abs/2009.10749).


## 1. Requirements

* Python 3.7
* Java 8 (or later)
  * Java environment variables set as described [here](https://pyjnius.readthedocs.io/en/stable/installation.html#installation)
* JAR-files ready (they should already be)
  * CPLEX (>=12.8.0): The file cplex.jar (for 12.10.0) is provided in the folder lib.
  * [SATS](http://spectrumauctions.org/) (>=0.7.0): The file sats-0.7.0.jar is provided in the folder lib.
* CPLEX Python API installed as described [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html)
* Make sure that your version of CPLEX is compatible with the cplex.jar file in the folder lib.

## 2. Dependencies

Prepare your python environment (whether you do that with `conda`, `virtualenv`, etc.) and enter this environment. You need to install Cython manually before to make sure the following command works.

Using pip:

```bash
pip install -r requirements.txt

```

## 3. Example installation using conda
```bash
conda create -n ica python=3.7
conda activate ica
conda install openjdk
pip install cython
pip install -r requirements.txt
```

## 4. How to run

### 4.1 Spectral Energy Analysis of Set Functions

Let $M:=\\\{1,\ldots,m\\\}$ denote the ground set (e.g., number of items in a combinatorial auctions), and let

$$\large v:\\\{0,1\\\}^m \to \mathbb{R}$$

denote the corresponding set function (e.g., a bidder's value functions for sets/bundles of items). Note that we identify the input sets here with their corresponding indicator vectors, e.g., for $M=\\\{1,2,3\\\}$ the set $A=\\\{1,3\\\}$ can be equivalently represented as indicator vector $x_A=(1,0,1)$, and thus we use as domain of $v$ $\\\{0,1\\\}^m$  instead of $2^M$ (i.e., the powerset of $M$).


Then the spectral energy of $v(\cdot)$ for cardinality $d\in M$ w.r.t to a set function Fourier transform $F$ is defined as

$$ \sum_{y\in \\\{0,1\\\}^m: |y| = d} \phi_{v}(y)^2 {\Huge /} \sum_{y\in \\\{0,1\\\}^m} \phi_{v}(y)^2$$

where $\phi_{v}(y)$ denotes the Fourier coefficient corresponding to $F$ at frequency $y$ (see [Fourier Analysis-based Iterative Combinatorial Auctions, Section 3 and Section 5.1](https://arxiv.org/abs/2009.10749)).


To analyze the spectral energy distribution of the set function $v:\\\{0,1\\\}^m\to \mathbb{R}$ of your choice go to the file ***analyze_spectral_energy.py***, set the size of the ground set $m$ and define your set function $v$ as a python function which gets as input a list of length $m$, i.e., below, instead of "..." enter the desired parameters and uncomment.


```python
# %% Define Set Function v: 2^M -> R_+ for spectral analysis

    # %% Define Set Function v: 2^M -> R_+ for spectral analysis

    # YOUR SET FUNCTION v: (uncomment if you use your own set function)
    # ---------------------------------------------------------------------------------------
    # m = ...
    # def v(x):
    #    return ...
    # ---------------------------------------------------------------------------------------

    # evaluate v at empty and full bundle (i.e, set).
    print('\n\nCheck set function v:')
    print(f'v((0,...,0))) = {v([0]*m):6.2f}')
    print(f'v((1,...,1))) = {v([1]*m):.2f}')

```

Note that as an example, we provide various bidders' value functions $v$ from the spectrum auction test suite (SATS). To define a bidder's value function, choose below the ***sats_value_model***, the ***sats_bidder_type*** and the ***seed***. If you define your own set function as outlined above, you need to comment this part.

```python
# %% Define Set Function v: 2^M -> R_+ for spectral analysis

    # YOUR SET FUNCTION v: (uncomment if you use your own set function)
    # ---------------------------------------------------------------------------------------
    # m = ...
    # def v(x):
    #    return ...
    # ---------------------------------------------------------------------------------------

    # EXAMPLE LSVM (or GSVM) NATIONAL (or REGIONAL) BIDDER (comment if you use your set function):
    # ---------------------------------------------------------------------------------------
    # For this example we use as set function v the national bidder from the SATS domain LSVM.

    # SELECT **********************
    sats_value_model = 'LSVM' # select from 'GSVM' and 'LSVM'
    sats_bidder_type = 'national' # select from 'regional' and 'national'
    seed = 1
    # *****************************
    if sats_value_model == 'LSVM':
        SATS_auction_instance = PySats.getInstance().create_lsvm(seed=seed, isLegacyLSVM=True)
        bidder_type_to_id_mapping = {'regional':[1,2,3,4,5],'national':[0]}
        bidder_id = random.sample(bidder_type_to_id_mapping[sats_bidder_type],1)[0]
    elif sats_value_model == 'GSVM':
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=seed, isLegacyGSVM=True)
        bidder_type_to_id_mapping = {'regional':[0,1,2,3,4,5],'national':[6]}
        bidder_id = random.sample(bidder_type_to_id_mapping[sats_bidder_type],1)[0]
    else:
        raise NotImplementedError(f'sats_value_model:{sats_value_model}')

    m = len(SATS_auction_instance.get_good_ids()) # number of items (size pof ground set); works up to m=32
    
    # create set function v, which gets as input a indicator list of size m, representing the set, and outputs a real number.
    def v(x):
        return SATS_auction_instance.calculate_value(bidder_id=bidder_id,
                                                     goods_vector=x)

    # ---------------------------------------------------------------------------------------

    # evaluate v at empty and full bundle (i.e, set).
    print('\n\nCheck set function v:')
    print(f'v((0,...,0))) = {v([0]*m):6.2f}')
    print(f'v((1,...,1))) = {v([1]*m):.2f}')
```

Once the set function $v$ is defined, run

```bash
$ python analyze_spectral_energy.py
```
to analyze the spectral energy distribution of $v$ (see [Fourier Analysis-based Iterative Combinatorial Auctions, Figure 1](https://arxiv.org/abs/2009.10749)).

This will create the following console output:

```bash
Check set function v:
v((0,...,0))) =   0.00
v((1,...,1))) = 489.14

1. creating indicators...
elapsed time: 0d 0h:0m:2s (14:42 21-06-2022)

2. calculate full set function v...
elapsed time: 0d 0h:1m:28s (14:44 21-06-2022)

3. calculate Fourier transforms...
calculate WHT...
elapsed time: 0d 0h:0m:2s (14:44 21-06-2022)
calculate FT3...
elapsed time: 0d 0h:0m:1s (14:44 21-06-2022)
calculate FT4...
elapsed time: 0d 0h:0m:1s (14:44 21-06-2022)

4. saving Fourier transforms...

5. plot spectral energy distribution...
saved as spectral_analysis-energy-distribution-plot.pdf)

```
and it will save

1. the spectral energy distribution plot as ***spectral_analysis-energy-distribution-plot.pdf***
2. the elapsed times fo the algorithm's steps as ***spectral_analysis-elapsed-times.json***
3. the computed result dictionary as ***spectral_analysis-results.pkl***

```python
results_dict = {'m':m,
                'seed':seed,
                'v_vec':v_vec,
                'v_wht':v_wht,
                'v_ft3':v_ft3,
                'v_ft4':v_ft4}
```

where ***v_vec*** is the vector of size $2^m$ representing the (whole) set function $v$ and ***v_wht***, ***v_ft3***, and ***v_ft4*** are the corresponding vectors representing the Fourier transforms WTH, FT3, and FT4 of $v$ (see [Fourier Analysis-based Iterative Combinatorial Auctions, Section 3](https://arxiv.org/abs/2009.10749)).

##### IMPORTANT
Note that this software currently only works up to $m\approx30$. Calculating the exact Fourier transforms for $m\>30$ requires using sparse FT algorithms instead.

### 4.2 Iterative Combinatorial Auction Mechanisms

To run any of the three hybrid mechanisms: HYBRID, HYBRID-FR, or HYBRID-FR-FA use
```bash
$ python simulation_hybrid_auctions.py
```
and select the desired world and auction mechanism. 

To run MLCA use 
```bash
$ python simulation_mlca.py
```

## Acknowledgements 

The algorithms for computing FT3 and FT4 sparse approximations was provided to us by the authors of [1].
The algorithm for computing WHT sparse approximations was provided to us by the authors of [2].

[1] Learning Set Functions that are Sparse in Non-Orthogonal Fourier Bases https://arxiv.org/abs/2010.00439

[2] Efficiently Learning Fourier Sparse Set Functions https://papers.nips.cc/paper/2019/hash/c77331e51c5555f8f935d3344c964bd5-Abstract.html

## Contact

Maintained by

Jakob Weissteiner (weissteiner)<br />
Website: www.jakobweissteiner.com<br />
E-mail: weissteiner@ifi.uzh.ch<br />



Chris Wendler (wendlerc)<br />
E-mail: chris.wendler@inf.ethz.ch
