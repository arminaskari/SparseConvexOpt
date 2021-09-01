Sparse Convex Optimization
=========

# Overview
This package solves the Sparse Programming problem described in [the paper](https://arxiv.org/pdf/2102.06742.pdf). The aforementioned paper solves \\( \ell_0 \\) constrained or penalized optimization problem. The algorithm in the paper works for non-convex objective functions \\(f \\) but this repo only implements linear regression and logistic regression loss functions. 

All the package dependencies are included in 'requirements.txt'

# Sample Usage
The jupyter notebook `sample_usage.ipynb` walks through a simple case of how to set up a problem and how to solve it.

# Experiment reproduction
```synthetic_expt.py``` is the code used to generate synthetic problems and solve them via the procedure described in the paper. To run `synthetic_expt.py` script, type
```
python synthetic_expt.py --dim 100,30,10 --loss l2 --l2reg constr --l0reg constr --sparsity 0.1 --gamma 1 --lowrank soft --iterate_over rank --param_grid 1,5,10  --save
```

Look at the module docstring for more information.


