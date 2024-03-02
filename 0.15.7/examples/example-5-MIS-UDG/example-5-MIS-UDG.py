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
# <a class=md-button href="example-5-MIS-UDG.py" download> Download Script </a>
# <a class=md-button href="../../assets/data/MIS-UDG-job.json" download> Download Job </a>
#
# <div class="admonition warning"> 
# <p class="admonition-title">Job Files for Complete Examples</p>
# <p>
# To be able to run the complete examples without having to submit your program to hardware and wait, you'll
# need to download the associated job files. These files contain the results of running the program on 
# the quantum hardware. 
#
# You can download the job files by clicking the "Download Job" button above. You'll then need to place
# the job file in the `data` directory that was created for you when you ran the `import` part of the script 
# (alternatively you can make the directory yourself, it should live at the same level as wherever you put this script).
# </p> 
# </div>
#

# %% [markdown]
# # Solving the Maximal Independent Set Problem on defective King Graph
# ## Introduction
# In this tutorial, we show how to use some of Bloqade's built-in tools to generate a
# defects in a graph and then use Bloqade to solve the Maximal Independent Set (MIS)
# problem on a Unit Disk Graph (UDG), which is easily expressable on Neutral Atom
# Hardware via the Rydberg blockade mechanism. We will not cover hybrid quantum-
# classical algorithms in this tutorial, but instead, we will use a simple parameter
# scan to find the optimal detuning value for an adiabatic ramp. We will cover hybrid
# quantum-classical algorithms in a future tutorial.

# %% [markdown]
# ## Define the Program.
# To define random defects on any Bloqade geometry, simply call the `add_defect_density`
# or `add_defect_count` methods on the geometry object. The `add_defect_density` method
# takes a float between 0 and 1 and uses that as the probability of a site being a
# defect. The `add_defect_count` method takes the number of defects to add to the
# geometry placed in random locations. Both ways take an  optional `rng` argument,
# a numpy random number generator. If no `rng` argument is provided, then the default
# numpy random number generator is used. Using the random number generator allows you
# to set the seed for reproducibility. After that, defining the pulse sequence is the
# same as in the previous tutorials.

# %%
from bloqade import load, save
from bloqade.atom_arrangement import Square
import numpy as np
import os
import matplotlib.pyplot as plt

if not os.path.isdir("data"):
    os.mkdir("data")

# setting the seed
rng = np.random.default_rng(1234)

durations = [0.3, 1.6, 0.3]

mis_udg_program = (
    Square(15, lattice_spacing=5.0)
    .apply_defect_density(0.3, rng=rng)
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, [0.0, 15.0, 15.0, 0.0])
    .detuning.uniform.piecewise_linear(
        durations, [-30, -30, "final_detuning", "final_detuning"]
    )
)

mis_udg_job = mis_udg_program.batch_assign(final_detuning=np.linspace(0, 80, 41))

# %% [markdown]
# ## Run On Hardware
# We can't run on our emulators because the program size is too large. Instead
# we will run on hardware.

# %%
filename = os.path.join(os.path.abspath(""), "data", "MIS-UDG-job.json")

if not os.path.isfile(filename):
    hw_batch = mis_udg_job.braket.aquila().run_async(shots=100)
    save(hw_batch, filename)

# %% [markdown]
# ## Plot Results
# Here, the total number of Rydberg excitations is plotted as a function of the final
# detuning. The total number of Rydberg excitations is a proxy for the largest
# independent set size because the number of violations to the Rydberg blockade is and
# will not scale with the size of the independent set. We start by loading the results

# %%

batch = load(filename)
# batch.fetch()
# save(filename, batch)

# %% [markdown]
# The report object already has a method to calculate the Rydberg densities. We can
# use this to calculate the average total Rydberg density for each final detuning.
# then, we can plot the results.

# %%

report = batch.report()

average_rydberg_excitation = report.rydberg_densities(filter_perfect_filling=False).sum(
    axis=1
)
final_detunings = report.list_param("final_detuning")

plt.plot(final_detunings, average_rydberg_excitation, color="#6437FF")
plt.xlabel("final detuning (rad/µs)")
plt.ylabel("total rydberg excitations")
plt.show()
