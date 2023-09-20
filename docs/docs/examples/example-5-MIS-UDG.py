# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     hide_notebook_metadata: false
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---


# %% [markdown]
# # Solving the Maximal Independent Set Problem on defective King Graph
# ## Introduction
# In this tutorial we show how to use some of Bloqade's built-in tools to generate a
# defects in a graph and then use Bloqade to solve the Maximal Independent Set (MIS)
# problem on a Unit Disk Graph (UDG) which are easily expressable on Neutral Atom
# Hardware via the Rydberg blockade mechanism. We will not cover hybrid quantum-
# classical algorithms in this tutorial, but instead we will use a simple parameter
# scan to find optimal detuning value for an adiabatic ramp. We will cover hybrid
# quantum-classical algorithms in a future tutorials.

# %% [markdown]
# ## Define the program.
# To define random defects on any Bloqade geometry simply call the `add_defect_density`
# or `add_defect_count` methods on the geometry object. The `add_defect_density` method
# takes a float between 0 and 1 and use that as the probability of a site being a 
# defect. The `add_defect_count` method takes an integer and uses that as the number of 
# defects to add to the geometry placed in random locations. Both methods take an 
# optional `rng` argument which is a numpy random number generator. If no `rng` argument
# is provided, then the default numpy random number generator is used. Using the random
# number generator allows you to set the seed for reproducibility. After that defining
# the pulse sequence is the same as in the previous tutorials.

# %%
from bloqade import load, save
from bloqade.atom_arrangement import Square
import numpy as np
import os
import matplotlib.pyplot as plt

# setting the seed
rng = np.random.default_rng(1234)

durations = [0.3, 1.6, 0.3]

mis_udg_program = (
    Square(15, 5.0)
    .apply_defect_density(0.3, rng=rng)
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, [0.0, 15.0, 15.0, 0.0])
    .detuning.uniform.piecewise_linear(
        durations, [-30, -30, "final_detuning", "final_detuning"]
    )
)

mis_udg_job = mis_udg_program.batch_assign(final_detuning=np.linspace(0, 80, 41))

# %% [markdown]
# ## Run On Hardware
# Here we can't run on our emulators because the proglem size is too large. Instead
# we will run on hardware.

filename = os.path.join(os.path.abspath(""), "data", "MIS-UDG-job.json")

if not os.path.isfile(filename):
    hw_batch = mis_udg_job.braket.aquila().run_async(shots=100)
    save(hw_batch, filename)

# %% [markdown]
# ## Plot Results
# Here the total number of Rydberg excitations is plotted as a function of the final
# detuning. The total number of Rydberg excitations are a proxy for the the largest
# independent set size because the number of violations to the Rydberg blockade is
# will not scalr with the size of the independent set. We start by loading the results

# %%

batch = load(filename)
# batch.fetch()
# save(filename, batch)

# %% [markdown]
# the report object already has a method to calculate the rydberg densities. We can
# ise this to calculate the average total rydberg density for each final detuning.
# then we can plot the results.

# %%

report = batch.report()

average_rydberg_excitation = report.rydberg_densities().sum(axis=1)
final_detunings = report.list_param("final_detuning")

plt.plot(final_detunings, average_rydberg_excitation)
plt.xlabel("final detuning (rad/µs)")
plt.ylabel("total rydberg excitations")
