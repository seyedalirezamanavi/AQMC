# AQMC
In this project, we used the Adiabatic Quantum Monte-Carlo Method to investigate the single-band Hubbard model.
A significant issue with the QMC-based algorithm is the so-called sign problem.
One solution to this problem is to increase the measurement rate.
Some of this code is written entirely in CUDA kernels to take advantage of the GPUs' massive parallelism.
This allows us to take measurements more frequently, resulting in lower temperatures. 
