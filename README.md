# Fourier Analysis-based Iterative Combinatorial Auctions

This is a piece of software used for computing the outcome of the HYBRID mechanisms and MLCA. The algorithms are described in detail in the attached paper.


## Requirements

* Python 3.6
* Java 8 (or later)
  * Java environment variables set as described [here](https://pyjnius.readthedocs.io/en/stable/installation.html#installation)
* JAR-files ready (they should already be)
  * CPLEX (>=12.8.0): The file cplex.jar (for 12.10.0) is provided in the folder lib.
  * [SATS](http://spectrumauctions.org/) (>=0.7.0): The file sats-0.7.0.jar is provided in the folder lib.
* CPLEX Python API installed as described [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html)
* Make sure that your version of CPLEX is compatible with the cplex.jar file in the folder lib.

## Dependencies

Prepare your python environment (whether you do that with `conda`, `virtualenv`, etc.) and enter this environment. You need to install Cython manually before to make sure the following command works.

Using pip:
```bash
$ pip install -r requirements.txt

```

## How to run

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
