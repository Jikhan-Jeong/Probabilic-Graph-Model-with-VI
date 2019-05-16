# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:07:33 2019

@author: Jikhan Jeong
"""

# download method for HPC

http://docs.pyro.ai/en/0.2.1-release/installation.html



# 2019_05_08_VI_pyro_intro

# pyro practice
# install (first) torch and (second) pyro required

#1. pytorch download: https://pytorch.org/get-started/locally/
#   (prompt) conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
#2. pyro download   : https://pyro.ai/
#   (prompt) pip install pyro-ppl

# reference: http://pyro.ai/examples/intro_part_i.html


import torch
import pyro

pyro.set_rng_seed(100)

# Primitive Stochastic Functions: X ~N(0,1)

loc =0.      # mean
scale =1.  # var
normal = torch.distributions.Normal(loc,scale)
x = normal.rsample()
print("sample",x)
print("log prob", normal.log_prob(x)) # score the sample from N(0,1)




# weather function with pytorch

def weather():
    cloudy = torch.distributions.Bernolli(0.3).sample() # bernolli distribution with prob = 0.3
    cloudy = 'cloudy' if cloudy.time() ==1.0 else 'sunny'
    mean_temp ={'cloudy':55.0, 'sunny':75.0}[cloudy]
    scale_temp ={'cloudy':10.0, 'sunny':15.0}[cloudy]
    temp = torch.dsitrubtions.Normal(mean_temp, scale_temp).resample
    return cloudy, temp.item()




# The pyro.sample Primitive
    
x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale))
x


# weather function with pyro

def weather_pyro():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp ={'cloudy':55.0, 'sunny':75.0}[cloudy]
    scale_temp ={'cloudy':10.0, 'sunny':15.0}[cloudy]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp,scale_temp))
    return cloudy, temp.item()
    
for _ in range(3):
    print(weather_pyro())


# Universality: Stochastic Recursion, Higher-order Stochastic Functions, and Random Control Flow
    
def ice_cream_sales():
    cloudy, temp = weather()    
    expected_sales = 200. if cloudy =='sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales,10.0))
    return ice_cream

# we can define a geometric distribution that counts the number of failures 
# until the first success like so:

def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t),pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1+geometric(p, t+1)

print(geometric(0.5))


def normal_product(loc, scale):
    z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
    y = z1 * z2
    return y

def make_normal_normal():
    mu_latent = pyro.sample("mu_latent", pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn

print(make_normal_normal()(1.))


    
    



