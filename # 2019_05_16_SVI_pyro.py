# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:27:38 2019

@author: Jikhan Jeong
"""

# 2019_05_16 An introduction to inference in pyro
# reference: https://pyro.ai/examples/intro_part_ii.html


import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)

# a simple example

def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))

conditioned_scale = pyro.condition(scale, data ={"measurement": 9.5})

def deferred_conditioned_scale(measurement, guess):
    return pyro.condition(scale, data ={"measurement": measurement})(guess)

def scale_obs(guess): # equivalent to conditioned scale above
    weight = pyro.sample("weight", dist.Normal(guess,1.))
    return pyro.sample("measurement", dist.Normal(weight, 1.), obs =9.5)

# Flexible Approximate Inference With Guide Functions (=posterior distribution)
# guide functions or guides, as approximate posterior distributions. 


def prefect_guide(guess):
    loc =(0.75**2* guess + 9.5) / (1+ 0.75**2)
    scale = np.sqrt(0.75**2/(1+ 0.75**2))
    return pyro.sample("weight", dist.Normal(loc, scale))


# Parametrized Stochastic Functions and Variational Inference
    
def intractable_scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(some_nonlinear_function(weight), 0.75))

simple_param_store = {}
a = simple_param_store.setdefault("a", torch.randn(1))

def scale_parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.))
    return pyro.sample("weight", dist.Normal(a, torch.abs(b))) # normal of sd should be postive, so abs

from torch.distributions import constraints

def scale_parametrized_guide_constrained(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.), constraint=constraints.positive)
    return pyro.sample("weight", dist.Normal(a, b))  # no more torch.abs

# stochastic variational inference

# 1. Parameters are always real-valued tensors
# 2. We compute Monte Carlo estimates of a loss function from samples of execution histories of the model and guide
# 3. We use stochastic gradient descent to search for the optimal parameters.
    
guess = 8.5
pyro.clear_param_store()

svi = pyro.infer.SVI(model = conditioned_scale,
                      guide = scale_parametrized_guide,
                      optim = pyro.optim.SGD({"lr":0.001, "momentum":0.1}),
                      loss=pyro.infer.Trace_ELBO())

losses, a, b = [], [] , []

num_steps = 2500

for t in range(num_steps):
    losses.append(svi.step(guess))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())



plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");
print('a = ',pyro.param("a").item())
print('b = ', pyro.param("b").item())



plt.subplot(1,2,1)
plt.plot([0,num_steps],[9.14,9.14], 'k:')
plt.plot(a)
plt.ylabel('a')

plt.subplot(1,2,2)
plt.ylabel('b')
plt.plot([0,num_steps],[0.6,0.6], 'k:')
plt.plot(b)
plt.tight_layout()
