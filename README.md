## PolyBranchX: Polymer representation through branched spatial processes to estimate network statistics
Generate shprtest path (SP) statistics of CGMD configurations characterized by crosslink density using the numerically estimated first-passage time (FPT) distribution of branched spatial processes.

## Cite
If you use the analytic results or the BRW code presented in this repository, cite
* Modeling Shortest Paths in Polymeric Networks using Spatial Branching Processes (https://doi.org/10.48550/arXiv.2310.18551)
```
@article{zhang2023modeling,
  title={Modeling Shortest Paths in Polymeric Networks using Spatial Branching Processes},
  author={Zhang, Zhenyuan and Mohanty, Shaswat and Blanchet, Jose and Cai, Wei},
  journal={arXiv preprint arXiv:2310.18551},
  year={2023}
}
```
More details on the anisotropic evolution of the polymer networks through the Monte Carlo BRW approach can be found in:
* On the First Passage Times of Branching Random Walks in Rd (https://arxiv.org/pdf/2404.09064)
```
@article{blanchet2024first,
  title={On the First Passage Times of Branching Random Walks in $$\backslash$mathbb R\^{} d$},
  author={Blanchet, Jose and Cai, Wei and Mohanty, Shaswat and Zhang, Zhenyuan},
  journal={arXiv preprint arXiv:2404.09064},
  year={2024}
}
```

Further analytic estimates and accelerated algorithms to reconstruct the SP distribution can be found in:
* Large Deviations of First Passage Times of Branching Random Walks in : Asymptotics and Algorithms (https://arxiv.org/pdf/2506.15072)
```
@article{blanchet2025large,
  title={Large Deviations of First Passage Times of Branching Random Walks in $$\backslash$mathbb $\{$R$\}$\^{} d $: Asymptotics and Algorithms},
  author={Blanchet, Jose and Cai, Wei and Mohanty, Shaswat and Zhang, Zhenyuan},
  journal={arXiv preprint arXiv:2506.15072},
  year={2025}
}
```

If you use the foundational arguments to motivate SP evolution as a microstructural parameter that goerns macroscopic material response then cite
* Topological origin of strain induced damage of multi-network elastomers by bond breaking (https://doi.org/10.1016/j.eml.2020.100883)
```
@article{yin2020topological,
  title={Topological origin of strain induced damage of multi-network elastomers by bond breaking},
  author={Yin, Yikai and Bertin, Nicolas and Wang, Yanming and Bao, Zhenan and Cai, Wei},
  journal={Extreme Mechanics Letters},
  volume={40},
  pages={100883},
  year={2020},
  publisher={Elsevier}
}
```
* Network evolution controlling strain-induced damage and self-healing of elastomers with dynamic bonds (https://arxiv.org/pdf/2401.11087)
```
@article{yin2024network,
  title={Network evolution controlling strain-induced damage and self-healing of elastomers with dynamic bonds},
  author={Yin, Yikai and Mohanty, Shaswat and Cooper, Christopher B and Bao, Zhenan and Cai, Wei},
  journal={Macromolecules},
  volume={57},
  number={13},
  pages={6410--6418},
  year={2024},
  publisher={ACS Publications}
}
```
* Why the strength of polymer networks is so low? (https://arxiv.org/pdf/2502.11339)
```
@article{mohanty2025strength,
  title={Why is the strength of a polymer network so low?},
  author={Mohanty, Shaswat and Blanchet, Jose and Suo, Zhigang and Cai, Wei},
  journal={arXiv preprint arXiv:2502.11339},
  year={2025}
}
```

## Installation
Use the following lines to get all the dependencies setup

```
git clone git@gitlab.com:micronano_public/PolyBranchX.git ; 
cd PolyBranchX ;
workdir=$(pwd) ;
python3 -m pip install -r py_requirements.txt ;
```

## Theoretical Estimates
To generate the reference $c_1$ results as a function of $\tilde{\lambda}$ for the branching random walk (BRW -- with delayed branching), branching Brownian motion (BBM) and gaussian branching random walk (GBRW)

```
cd $workdir/scripts ;
python3 theory_util.py --BRW --rho_range large ;
python3 theory_util.py --BBM --rho_range large ;
python3 theory_util.py --GBRW --rho_range large ;
```

## Numerical BRW and BBM 
* The script runs deterministic branching by default, but stochastic branching can be enabled (stationary process) by enabling the `--stationary` flag. 
* Delayed branching can be incorporated using the `--count_link` flag, whereas the termination from the finite polymer chain lenth can be nabled using `--terminate`. 
* The braching rate, $\tilde{\lmbda}$, can be specified using `--rate`, whereas the scaled jump length of $\sqrt{MSID(1/\tilde{\lambda})}$ can be assigned to `--jump`. 
* The correlation for the BMRW can be enabled using the `--correlated` flag.
* BBM and GBRW can be enabled using the `--gaussian` flag for gaussian steps instead of constant length jumps.
* The analysis for a signel branching rate can be computed using the `--single_analysis` flag. The mean SP as a function of various measuring lengths, $q_x$, can be computed using the `--tau_analysis` flag. The estimation of $c_1$ and $\overline{c}_1$ as a function of $\tilde{\lambda}$ can be computed using the `--crho_analysis` flag.
* Running the default 1000 paths might be time-consuming on the local system and you can choose smaller value, let's say, 50 by using the flag `--num_paths 50`

# Computing the SP distribution for a single branching rate for the BMRW, scaled BRW and the GBRW
Usage of the `--single_analysis` flag:
```
python3 branch_rw.py --count_link --single_analysis --plot_ref --rate $(bc -l <<< 0.0856) --dim 3 --jump $(bc -l <<< 1.34164) --stationary --purge --terminate;
python3 branch_rw.py --gaussian --count_link --single_analysis --plot_ref --rate $(bc -l <<< 0.0856) --dim 3 --jump $(bc -l <<< 1.34164) --stationary --purge --terminate --time_discrete 1;
python3 branch_rw.py --correlated --count_link --single_analysis --plot_ref --rate $(bc -l <<< 0.0856) --dim 3 --jump 1 --stationary --purge --terminate;
```

# Computing the mean SP at different measuring lengths
Usage of the `--tau_analysis` flag
```
python3 branch_rw.py --count_link --tau_analysis --box_min 20 --rate $(bc -l <<< 0.0856) --dim 3 --jump $(bc -l <<< 1.34164) --stationary --purge --terminate;
python3 branch_rw.py --gaussian --count_link --tau_analysis --box_min 20 --rate $(bc -l <<< 0.0856) --dim 3 --jump $(bc -l <<< 1.34164) --stationary --purge --terminate --time_discrete 1;
python3 branch_rw.py --correlated --count_link --tau_analysis --box_min 20 --rate $(bc -l <<< 0.0856) --dim 3 --jump 1 --stationary --purge --terminate;
```

# Computing the $\overline{c}_1$ as a function of branching rate
Usage of the `--crho_analysis` flag
```
python3 branch_rw.py --crho_analysis --box_min 20 --dim 3 --jump $(bc -l <<<1.34164) --rho_max $(bc -l <<<1.2) --rho_points 14 --purge  --stationary --count_link --terminate ;
python3 branch_rw.py --gaussian --crho_analysis --box_min 20 --dim 3 --jump $(bc -l <<<1.34164) --rho_max $(bc -l <<<1.2) --rho_points 14 --purge  --stationary --count_link --terminate --time_discrete 1;
python3 branch_rw.py --correlated --crho_analysis --box_min 20 --dim 3 --jump 1 --rho_max $(bc -l <<<1.2) --rho_points 14 --purge  --stationary --count_link --terminate ;
```

# Simulation the GBRW with an asymmetric jump distribution
A separate script `asymm_branch_rw.py` has been developed along the lines of the parent script `branch_rw.py`. The script is subjectoto further development but it currently supports the FPT calculations for a GBRW with an asymmetric jump distribution where the path purging criteria is modified to account for the resultant spatial distribution of the jumps. By default the script runs assuming asymmetric yet independent jumps, however, use the `--dependent` flag to introduce dependency between the directions for the asymmetric jumps. The covariance matrix for both cases is hard-coded for the time being.
```
python3 asymm_branch_rw.py --dependent --crho_analysis --box_min 20 --dim 3 --jump 1.732 --rho_max 2.9 --rho_points 19 --purge  --stationary --num_paths 1000 --gaussian ;
python3 asymm_branch_rw.py --dependent --crho_analysis --box_min 20 --dim 3 --jump 1.732 --rho_max 2.9 --rho_points 19 --purge  --stationary --num_paths 1000 --gaussian ;
```

## Publications 
* Modeling Shortest Paths in Polymeric Networks using Spatial Branching Processes (https://doi.org/10.48550/arXiv.2310.18551)

## Conference presentations
* To be added

## Support and Development

For any support regarding the implementation of the source code, contact the developers at: 
* Shaswat Mohanty (shaswatm@stanford.edu)
* Wei Cai (caiwei@stanford.edu)

For clarifications on the theoretical approach reach out to
* Zhenyuan Zhang (zzy@stanford.edu)
* Jose Blanchet (jose.blanchet@stanford.edu)


## Contributing
The development is actively ongoing and the sole contributors are Shaswat Mohanty and Zhenyuan Zhang.  Request or suggestions for implementing additional functionalities to the library can be made to the developers directly.

## Project status
Development is currently ongoing and is intended to be part of the dissertation project of Shaswat Mohanty and Zhenyuan Zhang.
