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
# # Nonequilibrium Dynamics of nearly Blockaded Rydberg Atoms
# ## Introduction
# In this example we will show how to generate multi-atom programs
# looking specifically at the dynamics of two atoms that are right
# on the blockade radius. First lsets start with the imports.

# %%

from bloqade import save, load
from bloqade.ir.location import Chain

import numpy as np
import matplotlib.pyplot as plt

import os

# %% [markdown]
# ## Program Definition
# We will start by defining a program. We set up a chain of two atoms
# with a parmaeterized distance between them. We then define a Rabi
# drive that is constant for a certain amount of time and then manually
# ramp down to 0.0. Given a `rabi_ampl` of 15 rad/µs the blockaded radius
# is 8.44 µm. We will look at the dynamics of the system for a distance
# of 8.5 µm to be every so slightly outside of the blockade radius.

# %%

initial_geometry = Chain(2, "distance")
program_waveforms = initial_geometry.rydberg.rabi.amplitude.uniform.piecewise_linear(
    durations=["ramp_time", "run_time", "ramp_time"],
    values=[0.0, "rabi_ampl", "rabi_ampl", 0.0],
)
program_assigned_vars = program_waveforms.assign(
    ramp_time=0.06, rabi_ampl=15, distance=8.5
)
batch = program_assigned_vars.batch_assign(run_time=0.05 * np.arange(31))
# %% [markdown]
# ## Run Emulator and Hardware
# To run the program on the emulator we can select the `braket` provider
# as a property of the `batch` object. Braket has its own emulator that
# we can use to run the program. To do this select `local_emulator` as
# the next option followed by the `run` method. Then we dump the results
# to a file so that we can use them later.

# %%
emu_filename = os.path.join(
    os.path.abspath(""), "data", "nonequilibrium-dynamics-blockade-emulation.json"
)
if not os.path.isfile(emu_filename):
    emu_batch = batch.braket.local_emulator().run(10000)
    save(emu_batch, emu_filename)

# %% [markdown]
# When running on the hardware we can use the `braket` provider as well.
# However, we will need to specify the `device` to run on. In this case
# we will use `Aquila` via the `aquila` method. Before that we must note 
# that because Aquila can support up to 256 atoms we need to make full use
# of the capabilities of the device. As we discussed in the previous examples
# we can use the `parallelize` which will allow us to run multiple copies of
# the program in parallel using the full user provided area of Aquila. This 
# has to be put before the `braket` provider. Then we dump the results
# to a file so that we can use them later.


# %%
filename = os.path.join(
    os.path.abspath(""), "data", "nonequilibrium-dynamics-blockade-job.json"
)

if not os.path.isfile(filename):
    hardware_batch = batch.parallelize(24).braket.aquila().run_async(shots=100)
    save(hardware_batch, filename)

# %% [markdown]
# ## Plotting the Results
# In order to show the complex dynamics of the system we will plot the
# probability of having `0`, `1`, or `2` Rydberg atoms as a function of time.
# We will do this for both the emulator and the hardware. We can use the 
# following function to get the probabilities from the shot counts of each 
# of the different configuration of the two Rydberg atoms: `00` `10`, `01`, `11`.
# Note that `0` corresponds to the Rydberg state while `1` corresponds to the
# ground state. as such `00` corresponds to two Rydberg atoms, `10`  and `01`
# corresponds to one Rydberg atom and one ground state atom, and `11` corresponds
# to two ground state atoms.

# %%

def rydberg_state_probabilities(shot_counts):
    probabilities_dict = {"0": [], "1": [], "2": []}

    # iterate over each of the task results
    for task_result in shot_counts:
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
# ## Extracting the counts amd probabilities
# We will now extract the counts and probabilities from the emulator and
# hardware runs. We will then plot the results. First we load the data
# from the files.

# %%

emu_batch = load(emu_filename)
emu_report = emu_batch.report()
emu_counts = emu_report.counts

hardware_batch = load(filename)
# hardware_batch.fetch() # uncomment to fetch results from Braket
# save(filename, hardware_batch)

# %% [markdown]
# To get the `counts` we need to get a `report` from the `batch` objects.
# Then with the report we can get the counts. The counts are a dictionary
# that maps the bitstring to the number of times that bitstring was measured.


# %%

emu_report = emu_batch.report()
hardware_report = hardware_batch.report()

emu_counts = emu_report.counts
hardware_counts = hardware_report.counts

emu_probabilities = rydberg_state_probabilities(emu_counts)
hw_probabilities = rydberg_state_probabilities(hardware_counts)

# %% [markdown]
#  plot 0, 1, and 2 Rydberg state probabilities but in separate plots

# %%
figure, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

emu_run_times = emu_report.list_param("run_time")
hardware_run_times = hardware_report.list_param("run_time")

axs[0].plot(emu_run_times, emu_probabilities["0"], marker=".", color="#878787")
axs[0].plot(hardware_run_times, hw_probabilities["0"], color="#6437FF", linewidth=4)
axs[0].title.set_text("0 Rydberg State")

axs[1].plot(emu_run_times, emu_probabilities["1"], marker=".", color="#878787")
axs[1].plot(hardware_run_times, hw_probabilities["1"], color="#6437FF", linewidth=4)
axs[1].title.set_text("1 Rydberg State")

axs[2].plot(
    emu_run_times,
    emu_probabilities["2"],
    marker=".",
    color="#878787",
    label="emulation",
)
axs[2].plot(
    hardware_run_times, hw_probabilities["2"], color="#6437FF", linewidth=4, label="qpu"
)
axs[2].title.set_text("2 Rydberg State")

axs[0].set_ylabel("Probability")
axs[2].legend()

figure.show()
