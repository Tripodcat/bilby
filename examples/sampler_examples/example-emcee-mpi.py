#!/usr/bin/env python
"""
An example of how to use bilby to perform paramater estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise

"""
from __future__ import division
import bilby
import numpy as np
import matplotlib.pyplot as plt
import sys

from mpi4py import MPI
from emcee.utils import MPIPool

# A few simple setup steps
sampler = 'emcee'
label = 'linear_regression_' + sampler
outdir = 'example'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll use standard Gaussian noise

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
sigma = np.random.normal(1, 0.01, N)
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

# We quickly plot the data to check it looks sensible
#fig, ax = plt.subplots()
#ax.plot(time, data, 'o', label='data')
#ax.plot(time, model(time, **injection_parameters), '--r', label='signal')
#ax.set_xlabel('time')
#ax.set_ylabel('y')
#ax.legend()
#fig.savefig('{}/{}_data.png'.format(outdir, label))

# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior
priors = dict()
priors['m'] = bilby.core.prior.Uniform(0, 5, 'm')
priors['c'] = bilby.core.prior.Uniform(-2, 2, 'c')


if MPI.COMM_WORLD.Get_size() == 1:
    sampler = bilby.sampler.Emcee(
            likelihood=likelihood, priors=priors, outdir=outdir,
            label=label, resume=False, nwalkers=100, nsteps=1000)
    result = sampler.run_sampler()
else:
    # mpi
    pool = MPIPool(loadbalance=True)
    from bilby.core import GLOBAL
    GLOBAL.sampler = bilby.sampler.Emcee(
            likelihood=likelihood, priors=priors, outdir=outdir,
            label=label, resume=False, nwalkers=100, nsteps=1000, pool=pool)
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    result = GLOBAL.sampler.run_sampler()

result.samples_to_posterior()
best_sample = result.posterior.iloc[result.posterior["log_likelihood"].idxmax()]
best_paras = best_sample[result.search_parameter_keys]
best_paras = best_paras.to_dict()
result.save_to_file()

result.plot_corner(parameters=best_paras)
result.plot_walkers()

# close pool
if MPI.COMM_WORLD.Get_size() != 1:
    pool.close()
