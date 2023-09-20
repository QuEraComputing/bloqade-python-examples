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
# In this example, we show how to use Bloqade to emulate the behavior of an adiabatic
# sweep on a pair of atoms, with the distance between atoms gradually increasing per
# task. As such, we will explore the Nature of the Rydberg interaction as the distance
# between atoms going from the non-interacting regime to the blockade regime. The
# cross-over between these two regimes will coincide with the blockade radius.


# %% [markdown]
# ## Defining the Program
# Now, we define our program of interest. As expected for an adiabatic protocol,
# we keep that Rabi frequency at a considerable value while slowly ramping the detuning 
# from a large negative to a positive value.
#
# Note that you can perform arithmetic operations directly on variables in the program
# but this requires the variable to be explicitly declared by passing a string to the
# `var` function and THEN doing arithmetic on it.

# %%
from bloqade import start, cast, var, save, load
import numpy as np

import matplotlib.pyplot as plt

import os

detuning_value = var("detuning_value")
durations = cast(["ramp_time", "run_time", "ramp_time"])
prog = (
    start.add_position([(0, 0), (0, "atom_distance")])
    .rydberg.rabi.amplitude.uniform.piecewise_linear(
        durations=durations, values=[0, "rabi_value", "rabi_value", 0]
    )
    .detuning.uniform.piecewise_linear(
        durations=durations,
        values=[
            -detuning_value,
            -detuning_value,
            detuning_value,
            detuning_value,
        ],
    )
)

distances = np.arange(4, 11, 1)
batch = prog.assign(
    ramp_time=1.0, run_time=2.0, rabi_value=15.0, detuning_value=15.0
).batch_assign(atom_distance=distances)

# %% [markdown]
# ## Run on Emulator and Hardware
# In previous examples, we have shown how to run a program on the emulator and hardware.
# First, we will run the program on the emulator and save the results to a file.

# %%
# get emulation batch, running 1000 shots per task
emu_filename = os.path.join(
    os.path.abspath(""), "data", "two-qubit-adiabatic-emulation.json"
)

if not os.path.isfile(emu_filename):
    emu_batch = batch.braket.local_emulator().run(1000)
    save(emu_batch, emu_filename)

# %% [markdown]
# Then, we can run the program on the hardware after parallelizing the tasks.
# We can then save the results to a file.

# %%

filename = os.path.join(os.path.abspath(""), "data", "two-qubit-adiabatic-job.json")

if not os.path.isfile(filename):
    hardware_batch = batch.parallelize(24).braket.aquila().submit(shots=100)
    save(hardware_batch, filename)


# %% [markdown]
# ## Plot the Results
# To show the blockade effect on the system, we will plot the
# probability of having `0`, `1`, or `2` Rydberg atoms as a function of time.
# We will do this for both the emulator and the hardware. We can use the
# following function to get the probabilities from the shot counts of each
# of the different configurations of the two Rydberg atoms: `00`, `10`, `01`, and `11`.
# Note that `0` corresponds to the Rydberg state while `1` corresponds to the
# ground state. As such, `00` corresponds to two Rydberg atoms, `10`  and `01`
# corresponds to one Rydberg atom and one ground-state atom, and `11` corresponds
# to two ground-state atoms.


# %%
def rydberg_state_probabilities(emu_counts):
    probabilities_dict = {"0": [], "1": [], "2": []}

    # iterate over each of the task results
    for task_result in emu_counts:
        # get total number of shots
        total_shots = sum(task_result.values())
        # get probability of each state
        probabilities_dict["0"].append(task_result.get("11", 0) / total_shots)
        probabilities_dict["1"].append(
            (task_result.get("10", 0) + task_result.get("01", 0)) / total_shots
        )
        probabilities_dict["2"].append(task_result.get("00", 0) / total_shots)

    return probabilities_dict


# %% [markdown]
# Before we can plot the results we need to load the data from the files.
# %%

# get emulation report and number of shots per each state
emu_batch = load(emu_filename)

# get hardware report and number of shots per each state
hardware_batch = load(filename)
# hardware_batch.fetch()
# save(hardware_batch, filename)


# %% [markdown]
# We can use the `rydberg_state_probabilities`
# function to extract the probabilities from the counts. This function
# takes a list of counts and returns a dictionary of probabilities for
# each state. The counts are obtained from the `report` of the `batch`
# object.
#
# Now, we can plot the results!

# %%

emu_report = emu_batch.report()
hardware_report = hardware_batch.report()

emu_rydberg_state_probabilities = rydberg_state_probabilities(emu_report.counts)
hw_rydberg_state_probabilities = rydberg_state_probabilities(hardware_report.counts)

emu_distances = emu_report.list_param("atom_distance")
hw_distances = hardware_report.list_param("atom_distance")

fig, ax = plt.subplots()
emu_colors = ["#55DE79", "#EDFF1A", "#C2477F"]  # Green, Yellow, Red

emu_lines = []
hw_lines = []
for rydberg_state, color in zip(["0", "1", "2"], emu_colors):
    (emu_line,) = ax.plot(
        emu_distances,
        emu_rydberg_state_probabilities[rydberg_state],
        label=rydberg_state + "-Rydberg",
        color=color,
    )
    (hw_line,) = ax.plot(
        hw_distances,
        hw_rydberg_state_probabilities[rydberg_state],
        color="#878787",
        label="QPU",
    )

    emu_lines.append(emu_line)
    hw_lines.append(hw_line)


ax.legend(handles=[*emu_lines, hw_lines[-1]])
ax.set_xlabel("time ($\mu s$)")
ax.set_ylabel("Probability")
fig.show()
