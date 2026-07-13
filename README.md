# PINN-TimeScales-IterativeBVP

This repository contains the numerical codes accompanying the paper

> Second-Order Iterative Boundary Value Problems on Time Scales:
> Identifiability, Stability, and PINN-Based Reconstruction

## Authors

1. Mahammad Khuddush

- Department of Mathematics,
  Vignan's Institute of Information Technology (Autonomous),
  Visakhapatnam, India

- School of Sciences,
  Woxsen University,
  Hyderabad, India

2. Prof. Jehad Alzabut
   Department of Applied Mathematics
   Prince Sultan University
   Saudi Arabia
## Contents

### Example 1
Continuous time scale T = R

- PINN reconstruction of unknown coefficients
- Identifiability verification
- Loss convergence
- Coefficient recovery plots

### Example 2
Discrete time scale T = {0,1,...,10}

- Delta-difference equation
- Coefficient identification
- PINN reconstruction

### Example 3
Hybrid time scale

T = [0,1) ∪ {2,3,...,10}

- Continuous-discrete dynamics
- Hybrid residual evaluation
- PINN coefficient reconstruction

## Requirements

Python 3.10+

PyTorch
NumPy
Matplotlib

## Citation

If you use these codes, please cite:

Khuddush, M.,
Second-Order Iterative Boundary Value Problems on Time Scales:
Identifiability, Stability, and PINN-Based Reconstruction.
