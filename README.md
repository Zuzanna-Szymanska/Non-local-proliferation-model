###########################################################################
###########################################################################

This folder contains files with codes for computations in the following papers:

[1] Z. Szymańska, J. Skrzeczkowski, B. Miasojedow, P. Gwiazda, "Bayesian inference of a non-local proliferation model", arXiv:2106.05955

[2] P. Gwiazda, B. Miasojedow, J. Skrzeczkowski, Z. Szymańska, "Convergence of the EBT method for a non-local model of cell proliferation with discontinuous interaction kernel", arXiv:2106.05115

Please cite exactly if you use this code:

@software{zuzanna_szymanska_2021_5565314,  
  author       = {Zuzanna Szymańska and Jakub Skrzeczkowski and Błażej Miasojedow and Piotr Gwiazda},  
  title        = {Non-local proliferation model},  
  year         = 2021,  
  publisher    = {Zenodo},  
  doi          = {10.5281/zenodo.5565314},  
  url          = {https://doi.org/10.5281/zenodo.5565314} 
}

###########################################################################
###########################################################################

File metropolis_hastings.py contains random walk Metropolis-Hastings algorithm used in [1].

File postprocessing.py contains auxiliary processing of output data generated by the metropolis-hastings.py program used in [1].

File folkman_a_b_c_time.py contains auxiliary code for solving the proposed non-local proliferation model used in [1].

File figures_Inverse_Proliferation.R is an auxiliary code used to create figures in [1].

File convergence_2D.py contains code for determining the order of convergence of the ETB-based method for a non-local model of 2D cellular colony dynamics with a discontinuous interaction kernel used in [2].

File convergence_3D.py contains code for determining the order of convergence of the ETB-based method for a non-local model of 3D cellular colony dynamics with a discontinuous interaction kernel used in [2].

File flatmetric.py contains extracted procedures for computing Wasserstein and flat metrics.
