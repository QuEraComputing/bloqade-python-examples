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
# # Two Qubit Adiabatic Sweep
# ## Introduction
# In this example we show how to use Bloqade to emulate the behavior
# of an adiabatic sweep on a pair of atoms, with the distance between
# atoms gradually increasing per task.
#
# First we import the necessary libraries

# %%
from bloqade import start, cast, var, save_batch, load_batch
import numpy as np

import matplotlib.pyplot as plt

import os

# %% [markdown]
# Now we define our program of interest. As expected for an adiabatic protocol
# we keep that Rabi frequency at a large value while slowly ramping the detuning from a large negative to positive value.
#
# Note that you can perform arithmetic operations directly on variables in the program but this requires 
# the variable to be explicitly declared by passing a string to the `var` function and THEN doing arithmetic on it.

# %%
durations = cast(["ramp_time", "run_time", "ramp_time"])
prog = (
    start.add_positions([(0, 0), (0, "atom_distance")])
    .rydberg.rabi.amplitude.uniform
    .piecewise_linear(durations=durations, values=[0, "rabi_value", "rabi_value", 0])
    .detuning.uniform.piecewise_linear(durations=durations, values=[-1 * var("detuning_value"), -1 * var("detuning_value"), "detuning_value", "detuning_value"])
    )

distances = np.around(np.arange(4, 11, 1), 13)
batch = prog.assign(
    ramp_time = 1.0,
    run_time = 2.0,
    rabi_value = 15.0,
    detuning_value = 15.0
).batch_assign(atom_distance = distances)

# %% [markdown]
# With our program fully defined (now considered a "batch" owing to the fact that the parameter sweep on the atom distance generates a "batch" of tasks)
# we have the ability to emulate it locally OR submit it to hardware.

# %%
# get emulation batch, running 1000 shots per task
emu_batch = batch.braket.local_emulator().run(1000)

# submit to HW, running 100 shots per task

filename = os.path.join(os.path.abspath(""), "data", "two-qubit-adiabatic.json")

if not os.path.isfile(filename):
    hardware_batch = batch.parallelize(24).braket.aquila().submit(shots=100)
    save_batch(filename, hardware_batch)

# %%

# get emulation report and number of shots per each state
emu_report = emu_batch.report()
emu_counts = emu_report.counts

# get hardware report and number of shots per each state
hardware_batch = load_batch(filename)
# hardware_batch.fetch()
# save_batch(filename, hardware_batch)
hardware_report = hardware_batch.report()
hardware_counts = hardware_report.counts

# %% [markdown]
#
# We define the following function to look at the number of shots associated with each possible number of 
# rydberg states and calculate the associated probabilities per each atom distance.

# %%
def rydberg_state_probabilities(emu_counts):

    probabilities_dict = { "0": [], "1": [], "2": []}

    # iterate over each of the task results
    for task_result in emu_counts:
        # get total number of shots
        total_shots = sum(task_result.values())
        # get probability of each state
        probabilities_dict["0"].append(task_result.get("11",0) / total_shots)
        probabilities_dict["1"].append((task_result.get("10",0) + task_result.get("01", 0)) / total_shots)
        probabilities_dict["2"].append(task_result.get("00",0) / total_shots)

    return probabilities_dict

# %% [markdown]
# Now we can plot the results!

# %%
emu_rydberg_state_probabilities = rydberg_state_probabilities(emu_counts)        
hw_rydberg_state_probabilities = rydberg_state_probabilities(hardware_counts)

fig, ax = plt.subplots()
emu_colors = ["#55DE79", "#EDFF1A", "#C2477F"] # Green, Yellow, Red
ax.set_xlabel("Distance ($\mu m$)")
ax.set_ylabel("Probability")
for rydberg_state, color in zip(emu_rydberg_state_probabilities, emu_colors):
    ax.plot(distances, emu_rydberg_state_probabilities[rydberg_state], label=rydberg_state + "Rydberg", color=color)

for rydberg_state in hw_rydberg_state_probabilities:
    ax.plot(distances, hw_rydberg_state_probabilities[rydberg_state], color="#878787")
ax.legend()

fig.show()