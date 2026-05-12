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
# <a class=md-button href="example-2-nonequilibrium-dynamics-blockade-radius.py" download> Download Script </a>
# <a class=md-button href="../../assets/data/nonequilibrium-dynamics-blockade-job.json" download> Download Job </a>
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
# # Nonequilibrium Dynamics of nearly Blockaded Rydberg Atoms
# ## Introduction
# In this example we will show how to generate multi-atom programs looking specifically
# at the dynamics of two atoms that are right on the blockade radius. First let's start
# with the imports.

# %%

import os

import numpy as np
import matplotlib.pyplot as plt
from bloqade.analog import load, save
from bloqade.analog.atom_arrangement import Chain

if not os.path.isdir("data"):
    os.mkdir("data")

# %% [markdown]
# ## Program Definition
# We will start by defining a program. We set up a chain of two atoms
# with a parameterized distance between them. We then define a Rabi
# like in the original Rabi oscillation example. Given a Rabi amplitude of 15 rad/µs,
# the blockade radius is 8.44 µm. We choose an atom distance of 8.5 µm so the pair is
# just outside that radius. This puts the interaction strength on the same scale as the
# Rabi drive, making it a useful regime for seeing the crossover between independent
# Rabi oscillations and blockade-limited dynamics.
# %%

ramp_time = 0.06
rabi_amplitude = 15
atom_distance = 8.5
run_times = 0.05 * np.arange(31)

initial_geometry = Chain(2, lattice_spacing="distance")
program_waveforms = initial_geometry.rydberg.rabi.amplitude.uniform.piecewise_linear(
    durations=["ramp_time", "run_time", "ramp_time"],
    values=[0.0, "rabi_ampl", "rabi_ampl", 0.0],
)
program_assigned_vars = program_waveforms.assign(
    ramp_time=ramp_time, rabi_ampl=rabi_amplitude, distance=atom_distance
)
batch = program_assigned_vars.batch_assign(run_time=run_times)

# %% [markdown]
# The blockade radius is defined by the length scale where the van der Waals
# interaction $V(R)=C_6/R^6$ matches the Rabi drive $\Omega$. For Aquila's Rydberg
# state, $C_6 = 2\pi \times 862690$ rad/µs µm$^6$. Setting $V(R_b)=\Omega$ gives
# $R_b = (C_6 / \Omega)^{1/6}$, so this program places the atoms just beyond the
# nominal blockade boundary.

# %%
aquila_c6 = 2 * np.pi * 862690
blockade_radius = (aquila_c6 / rabi_amplitude) ** (1 / 6)
interaction_strength = aquila_c6 / atom_distance**6

distance_values = np.linspace(6.5, 11.0, 200)
interaction_values = aquila_c6 / distance_values**6

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(distance_values, interaction_values, color="#6437FF", label="$V(R)$")
ax.axhline(rabi_amplitude, color="#878787", linestyle="--", label="$\\Omega$")
ax.axvline(blockade_radius, color="#C2477F", linestyle=":", label="$R_b$")
ax.scatter(
    [atom_distance],
    [interaction_strength],
    color="#C2477F",
    zorder=3,
    label="Program distance",
)
ax.set_xlabel("Atom distance (µm)")
ax.set_ylabel("Angular frequency (rad/µs)")
ax.set_title("Interaction Strength Near the Blockade Radius")
ax.legend()
plt.show()

# %% [markdown]
# The drive pulse ramps the Rabi amplitude up over 0.06 µs, holds it constant for the
# scanned run time, and then ramps back down to satisfy hardware constraints. The plot
# below shows the longest pulse in this sweep.

# %%
pulse_times = np.array(
    [0.0, ramp_time, ramp_time + run_times[-1], 2 * ramp_time + run_times[-1]]
)
pulse_amplitudes = np.array([0.0, rabi_amplitude, rabi_amplitude, 0.0])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(pulse_times, pulse_amplitudes, color="#6437FF")
ax.fill_between(pulse_times, pulse_amplitudes, color="#6437FF", alpha=0.15)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Rabi amplitude (rad/µs)")
ax.set_title("Rabi Pulse Schedule for the Longest Evolution")
plt.show()
# %% [markdown]
# ## Run Emulator and Hardware
# Once again we will run the emulator and hardware. We will use the
# `local_emulator` method to run the emulator locally. We will then
# save the results to a file so that we can use them later.

# %%
emu_filename = os.path.join(
    os.path.abspath(""), "data", "nonequilibrium-dynamics-blockade-emulation.json"
)
if not os.path.isfile(emu_filename):
    emu_batch = batch.bloqade.python().run(10000)
    save(emu_batch, emu_filename)

# %% [markdown]
# When running on the hardware we will also parallelize the batch and submit.
#
# <div class="admonition danger">
# <p class="admonition-title">Hardware Execution Cost</p>
# <p>
#
# For this particular program, 31 tasks are generated with each task having 100 shots, amounting to
#  __USD \\$40.30__ on AWS Braket.
#
# </p>
# </div>

# %%
filename = os.path.join(
    os.path.abspath(""), "data", "nonequilibrium-dynamics-blockade-job.json"
)

if not os.path.isfile(filename):
    hardware_batch = batch.parallelize(24).braket.aquila().run_async(shots=100)
    save(hardware_batch, filename)


# %% [markdown]
# ## Plotting the Results
# In order to show the complex dynamics we will plot the probability of having `0`, `1`
# , or `2` Rydberg atoms as a function of time. We will do this for both the emulator
# and the hardware. We can use the `rydberg_state_probabilities` function to extract
# the probabilities from the counts. This function takes a list of counts and returns a
# dictionary of probabilities for each state. The counts are obtained from the `report`
# of the `batch` object.
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
# ## Extracting the counts and probabilities
# We will now extract the counts and probabilities from the emulator and hardware runs.
# We will then plot the results. First we load the data from the files.

# %%

emu_batch = load(emu_filename)
emu_report = emu_batch.report()
emu_counts = emu_report.counts()

hardware_batch = load(filename)
# hardware_batch.fetch() # uncomment to fetch results from Braket
# save(filename, hardware_batch)

# %% [markdown]
# To get the `counts` we need to get a `report` from the `batch` objects. Then with the
# report we can get the counts. The counts are a dictionary that maps the bitstring to
# the number of times that bitstring was measured.


# %%

emu_report = emu_batch.report()
hardware_report = hardware_batch.report()


emu_rydberg_state_probabilities = rydberg_state_probabilities(emu_report.counts())
hw_rydberg_state_probabilities = rydberg_state_probabilities(hardware_report.counts())

# %% [markdown]
#  plot 0, 1, and 2 Rydberg state probabilities but in separate plots

# %%
figure, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

emu_run_times = emu_report.list_param("run_time")
hardware_run_times = hardware_report.list_param("run_time")

emu_colors = ["#55DE79", "#EDFF1A", "#C2477F"]  # Green, Yellow, Red

emu_lines = []
hw_lines = []
for ax, rydberg_state, color in zip(axs, ["0", "1", "2"], emu_colors):
    (hw_line,) = ax.plot(
        hardware_run_times,
        hw_rydberg_state_probabilities[rydberg_state],
        label=rydberg_state + "-Rydberg QPU",
        color=color,
    )
    (emu_line,) = ax.plot(
        emu_run_times,
        emu_rydberg_state_probabilities[rydberg_state],
        color="#878787",
    )
    emu_line.set_label("Emulator")

    emu_lines.append(emu_line)
    hw_lines.append(hw_line)

    ax.set_xlabel("time ($\mu s$)")
    ax.set_ylabel("Probability")

ax.legend(handles=[*hw_lines, emu_lines[-1]])

plt.show()

# %% [markdown]
# We can compress the same data into the mean Rydberg density,
# $\langle n_r \rangle = (P_1 + 2P_2)/2$, which gives the average excited-state
# population per atom. Near the blockade radius, the two-atom state is suppressed, so
# the density does not simply follow the single-atom Rabi oscillation curve.

# %%
emu_mean_rydberg_density = (
    np.array(emu_rydberg_state_probabilities["1"])
    + 2 * np.array(emu_rydberg_state_probabilities["2"])
) / 2
hw_mean_rydberg_density = (
    np.array(hw_rydberg_state_probabilities["1"])
    + 2 * np.array(hw_rydberg_state_probabilities["2"])
) / 2

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    emu_run_times,
    emu_mean_rydberg_density,
    color="#878787",
    marker=".",
    label="Emulator",
)
ax.plot(
    hardware_run_times,
    hw_mean_rydberg_density,
    color="#6437FF",
    marker=".",
    label="QPU",
)
ax.set_xlabel("time ($\\mu s$)")
ax.set_ylabel("Mean Rydberg density")
ax.legend()
plt.show()
