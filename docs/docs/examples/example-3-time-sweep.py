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
# # 1D Z2 State Preparation
# ## Introduction
# In this example we show how to create the Z2 ordered phase on a 1D chain of atoms and
# how to perform a scan over the sweep time to understand the behavior of an adiabatic
# pulse schedule in its formation.

# %% [markdown]
# Let's import all the tools we'll need.

# %%
from bloqade import save, load
from bloqade.atom_arrangement import Chain
import numpy as np
import os
import matplotlib.pyplot as plt

# %% [markdown]
# ## Program Definition We define a program where our geometry is a chain of 11 atoms
# with a distance of 6.1 micrometers between atoms.

# The pulse schedule presented here should be reminiscent of the Two Qubit Adiabatic
# Sweep example although we've opted to reserve variable usage for values that will
# actually have their parameters swept.

# %%

# Define relevant parameters for the lattice geometry and pulse schedule
n_atoms = 11
lattice_const = 6.1
min_time_step = 0.05

# Define Rabi amplitude and detuning values.
# Note the addition of a "sweep_time" variable
# for performing sweeps of time values.
rabi_amplitude_values = [0.0, 15.8, 15.8, 0.0]
rabi_detuning_values = [-16.33, -16.33, 16.33, 16.33]
durations = [0.8, "sweep_time", 0.8]

time_sweep_z2_prog = (
    Chain(n_atoms, lattice_const)
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude_values)
    .detuning.uniform.piecewise_linear(durations, rabi_detuning_values)
)

# Allow "sweep_time" to assume values from 0.05 to 2.4 microseconds for a total of
# 20 possible values.
# Starting at exactly 0.0 isn't feasible so we use the `min_time_step` defined
# previously.
time_sweep_z2_job = time_sweep_z2_prog.batch_assign(
    sweep_time=np.linspace(min_time_step, 2.4, 20)
)

# %% [markdown]
# ## Running on the Emulator and Hardware
# With our program properly composed we can now easily send it off to both the emulator
# and hardware.

# We select the Braket emulator and tell it that for each variation of the "time_sweep"
# variable we'd like to run 10000 shots. For the hardware we take advantage of the fact
# that 11 atoms takes up so little space on the machine we can duplicate that geometry
# multiple times to get more data per shot. We set a distance of 24 micrometers between
# copies to minimize potential interactions between them.

# For both cases, to allow us to submit our program without having to wait on immediate
# results from hardware (which could take a while considering queueing and window
# restrictions), we save the necessary metadata to a file that can then be reloaded
# later and results fetched when they are available.

# %%

emu_filename = os.path.join(os.path.abspath(""), "data", "time-sweep-emulation.json")
if not os.path.isfile(emu_filename):
    emu_future = time_sweep_z2_job.braket.local_emulator().run(shots=10000)
    save(emu_future, emu_filename)

filename = os.path.join(os.path.abspath(""), "data", "time-sweep-job.json")
if not os.path.isfile(filename):
    future = time_sweep_z2_job.parallelize(24).braket.aquila().run_async(shots=100)
    save(future, filename)

# %% [markdown]
# ## Plotting the Results To make our lives easier we define a trivial function to
# extract the probability of the Z2 phase from each of the tasks generated from the
# parameter sweep. The counts are obtained from the `report`of the batch object.

# %%


def get_z2_probabilities(report):
    z2_probabilities = []

    for count in report.counts:
        z2_probability = count["01010101010"] / sum(list(count.values()))
        z2_probabilities.append(z2_probability)

    return z2_probabilities


# %% [markdown]
# ## Extracting Counts And ProbabilitiesWe will now extract the counts and probabilities
# from the emulator and hardware runs. We will then plot the results. First we load the
# data from the files:

# %%
# retrieve results from HW
emu_batch = load(emu_filename)
hardware_batch = load(filename)

# Uncomment lines below to fetch results from Braket
# hardware_batch = hardware_batch.fetch()
# save(hardware_batch, filename)

# %% [markdown]

# To get the counts we need to get a report from the batch objects. Then with the
# report we can get the counts. The counts are a dictionary that maps the bitstring to
# the number of times that bitstring was measured.

# %%
emu_report = emu_batch.report()
hardware_report = hardware_batch.report()
emu_probabilities = get_z2_probabilities(emu_report)
hardware_probabilities = get_z2_probabilities(emu_report)

emu_sweep_times = emu_report.list_param("sweep_time")
hardware_sweep_times = hardware_report.list_param("sweep_time")


plt.plot(emu_sweep_times, emu_probabilities, label="Emulator")
plt.plot(hardware_sweep_times, hardware_probabilities, label="Hardware")
plt.show()

# %% [markdown]
## Analysis
# As expected, we see that if we allow the pulse schedule to run for a longer
# and longer period of time (more "adiabatically") we have an increasing
# probability of creating the Z2 phase.
