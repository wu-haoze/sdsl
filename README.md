# Self-Driven Strategy-Learning

This repository contains the Bounded Model Checking implementation described in the paper "Lightweight Online Learning for Sets of Related 
Problems in Automated Reasoning"

The implementation here can be used to reproduce the results in the paper.

## Requirements

The implementation is tested on **python3.8.10**, 
but should work on more recent python versions.

### Python packages
The Python package dependencies can be installed by

``pip install -r requirements.txt``

### Solvers

The implementation also depends on binaries of several constraint solvers. 
**If you are using Linux, chances are you do not need to install anything, 
as the static binaries are provided in ``bins/``.**

Otherwise, you need to download and build the following tools, and move
the binaries to the ``bins/`` directory:

| Solver | Version |
| ------ | ------- |
| kissat | https://github.com/arminbiere/kissat/commit/97917ddf2b12adc6f63c7b2a5a403a1ee7d81836 |
| pono | https://github.com/anwu1219/pono/commit/8157c1cd79cbdbfca59a65011e1a61b61c79a9f8 |
| boolector | https://github.com/anwu1219/pono/commit/ed11ef3ebcffd4966e231be82ea0c7d8da77df2b |
| cadical | https://github.com/arminbiere/cadical/tree/a5f15211db36c3956764e18194dd5bd63bf3b5e6 |

The installation process of those tools are described in their READMEs.

# How to run

The BMC procedure can be run on the command line via``scripts/run.py``. It takes in
BMC problems in **btor/btor2** format.

For example, one can use the following command
to run BMC with Self-driven Strategy-Learning (sdsl) on a benchmark
*benchmarks/all_unknown/arbitrated_top_n2_w128_d64_e0.btor2* with a sampling budget of 2 minutes:

``./scripts/run.py benchmarks/all_unknown/arbitrated_top_n2_w128_d64_e0.btor2 --sampling-budget 120``

To run it without sdsl:

``./scripts/run.py benchmarks/all_unknown/arbitrated_top_n2_w128_d64_e0.btor2 --mode bmc``

For debugging purpose and also for examining the effect of sdsl in real time, we also added a **shadow** mode which 
will solve each query using the default solving strategy as well as the learned solving strategy and print out
the runtime comparison. This mode can be triggered by the ``--shadow`` flag:
``./scripts/run.py benchmarks/all_unknown/arbitrated_top_n2_w128_d64_e0.btor2  --sampling-budget 120 --shadow``

As described in the paper, there are several configurable input parameters.
Instructions for how to set them can be obtained by:

``./scripts/run.py --help``

The default input parameters are what is used in Table II of the paper. 
