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
# <a class=md-button href="example-3-2d-ordered-state.py" download> Download Script </a>
# <a class=md-button href="../../assets/data/striated-phase-hardware.json" download> Download Job </a>
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
# # 2D State Preparation
# ## Introduction
# In this example we show how to create the Striated Phase
# on a 2D chain of atoms.

# %% [markdown]
# You might notice that the tools we need to import are
# a lot shorter than prior instances. This is because
# we're taking advantage of bloqade Python's built-in
# visualization capabilities instead of crafting
# a new plot with matplotlib.

# %%
from bloqade.atom_arrangement import Square
from bloqade import save, load
from bokeh.io import output_notebook

import os

if not os.path.isdir("data"):
    os.mkdir("data")

# This tells Bokeh to display output in the notebook
# versus opening a browser window
output_notebook()

# %% [markdown]
# ## Program Definition
# We define a program where our geometry is a square lattice of 3x3 atoms. Notice that
# unlke the 1D Z2 state preparation example the detuning now ramps to a higher value
# and the atoms are closer together.
# %%
# Have atoms separated by 5.9 micrometers
L = 3
lattice_const = 5.9

rabi_amplitude_values = [0.0, 15.8, 15.8, 0.0]
rabi_detuning_values = [-16.33, -16.33, "delta_end", "delta_end"]
durations = [0.8, "sweep_time", 0.8]

prog = (
    Square(L, lattice_const)
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude_values)
    .detuning.uniform.piecewise_linear(durations, rabi_detuning_values)
)

batch = prog.assign(delta_end=42.66, sweep_time=2.4)

# %% [markdown]
# ## Submitting to Emulator and Hardware
# Just as in prior examples, we submit our program to both hardware and the emulator and
# save the intermediate data in files for convenient fetching when the results are ready from hardware,
# as well as avoiding having to repeat emulation runs for the purposes of analysis.

# Considering how small a 3 x 3 lattice of atoms is relative to machine capabilities,
# we also take advantage of parallelization to duplicate the geometry and get more
# data per shot when submitting to Hardware.

# %%
emu_filename = os.path.join(
    os.path.abspath(""), "data", "striated-phase-emulation.json"
)
if not os.path.isfile(emu_filename):
    emu_future = batch.braket.local_emulator().run(shots=10000)
    save(emu_future, emu_filename)

hw_filename = os.path.join(os.path.abspath(""), "data", "striated-phase-hardware.json")
if not os.path.isfile(hw_filename):
    future = batch.parallelize(24).braket.aquila().run_async(shots=100)
    save(future, hw_filename)

# %% [markdown]
# ## Extracting Results
# We can reload our files to get results:

# %%
# retrieve results from emulator and HW
emu_batch = load(emu_filename)
hardware_batch = load(hw_filename)

# Uncomment lines below to fetch results from Braket
# hardware_batch = hardware_batch.fetch()
# save(hardware_batch, filename)

# %% [markdown]
# ## Visualizing Results With Ease
# In prior examples we've leverage Bloqade's ability to automatically put
# hardware and emulation results into the formats we need to make analysis easier.
#
# Now we'll go one step further by letting Bloqade Python do the visualization for us.
# In this case we'll visualize the Rydberg Densities of our system overlaid on the original
# geometry with just the following:

# %%
emu_batch.report().show()

# %% [markdown]
# Just as before, we let Bloqade generate a `report` which contains all the results in
# easy to digest format but we invoke the `.show()` method of our report which us
# easily get an idea of the results of our experiment with interactive plots.

# The plot that mos interests us is the one on the right under the "Rydberg Density" section.

# %%
hardware_batch.report().show()

# %% [markdown]
# Considering Bloqade's goal of a uniform visualization pipeline, we can get the same
# ability for results from hardware. Note that we can confirm the program does what it's
# supposed to as results from emulation agree with those from hardware quite well.
