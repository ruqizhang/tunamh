# Asymptotically Optimal Exact Minibatch Metropolis-Hastings
This repository contains code for [Asymptotically Optimal Exact Minibatch Metropolis-Hastings](https://arxiv.org/abs/2006.11677), NeurIPS 2020, Spotlight

# Dependencies
* [Julia 1.4.0](https://julialang.org/)

## Experiments
To run TunaMH on logistic regression on MNIST,
```
cd logistic
julia tunamh.jl
```
To run other methods on this task, use `julia mh.jl` for standard MH, `julia flymc.jl` for FlyMC and `julia tfmh.jl` for Truncated Factorized MH. 