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
# like in the original Rabi oscillation example. Given a `rabi_ampl` of 15 rad/microsecond,
# the blockade radius is about 8.44 micrometers. We place the atoms at 8.5 micrometers,
# just outside that radius, so the pair can partially leave the fully blockaded regime.
# Sweeping `run_time` then shows the crossover between collective blockade dynamics and
# independent two-atom Rabi oscillations.
# %%

initial_geometry = Chain(2, lattice_spacing="distance")
program_waveforms = initial_geometry.rydberg.rabi.amplitude.uniform.piecewise_linear(
    durations=["ramp_time", "run_time", "ramp_time"],
    values=[0.0, "rabi_ampl", "rabi_ampl", 0.0],
)
program_assigned_vars = program_waveforms.assign(
    ramp_time=0.06, rabi_ampl=15, distance=8.5
)
run_times = 0.05 * np.arange(31)
batch = program_assigned_vars.batch_assign(run_time=run_times)

# %% [markdown]
# Before running the program, it is useful to compare the chosen atom separation with
# the blockade radius. Inside the blockade radius, the `|rr>` state is strongly shifted
# and double excitation is suppressed. Here the atoms sit slightly beyond the estimate,
# so the two-Rydberg probability is small at early times but can grow during the sweep.

# %%
atom_distance = 8.5
blockade_radius = 8.44

fig, ax = plt.subplots(figsize=(6, 2.5))
atom_positions = np.array([0.0, atom_distance])
ax.scatter(atom_positions, np.zeros_like(atom_positions), s=180, color="#6437FF")
for position in atom_positions:
    ax.axvspan(
        position - blockade_radius,
        position + blockade_radius,
        color="#6437FF",
        alpha=0.08,
    )

ax.plot(atom_positions, np.zeros_like(atom_positions), color="#878787", linewidth=2)
ax.annotate(
    f"{atom_distance:.2f} micrometers",
    xy=(atom_distance / 2, 0),
    xytext=(atom_distance / 2, 0.18),
    ha="center",
    arrowprops={"arrowstyle": "<->", "color": "#333333"},
)
ax.set_xlim(-1.0, atom_distance + 1.0)
ax.set_ylim(-0.3, 0.35)
ax.set_yticks([])
ax.set_xlabel("Position (micrometers)")
ax.set_title("Two atoms placed just outside the blockade radius")
plt.show()

# %% [markdown]
# The Rabi drive ramps on, stays flat for the selected `run_time`, and ramps off. The
# sweep repeats the same shape with different plateau durations. The plot below shows
# the longest pulse in the batch, which is the largest `run_time` used in the results.

# %%
ramp_time = 0.06
rabi_ampl = 15
longest_run_time = max(run_times)
times = np.array(
    [
        0.0,
        ramp_time,
        ramp_time + longest_run_time,
        2 * ramp_time + longest_run_time,
    ]
)
amplitudes = np.array([0.0, rabi_ampl, rabi_ampl, 0.0])

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(times, amplitudes, color="#6437FF", linewidth=3)
ax.fill_between(times, amplitudes, color="#6437FF", alpha=0.12)
ax.set_xlabel("Time ($\\mu s$)")
ax.set_ylabel("Rabi amplitude (rad/$\\mu s$)")
ax.set_title("Representative pulse schedule")
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
#  Plot 0, 1, and 2 Rydberg state probabilities but in separate plots. Comparing these
#  traces shows how much of the population stays in the ground-state manifold, how much
#  reaches one Rydberg excitation, and when the double-excitation channel becomes visible.

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

    ax.set_xlabel("time ($\\mu s$)")
    ax.set_ylabel("Probability")

ax.legend(handles=[*hw_lines, emu_lines[-1]])

plt.show()

# %% [markdown]
# A complementary view is the site-resolved Rydberg density. Since the two atoms are
# separated by almost one blockade radius, the density remains nearly symmetric between
# sites, while differences between the emulator and hardware make the finite-shot and
# device effects easier to see than in the summed probabilities alone.

# %%
def rydberg_site_densities(report):
    return np.array([1 - bitstrings.mean(axis=0) for bitstrings in report.bitstrings()])


emu_density = rydberg_site_densities(emu_report)
hw_density = rydberg_site_densities(hardware_report)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, density, sweep_times, title in [
    (axs[0], emu_density, emu_run_times, "Emulator"),
    (axs[1], hw_density, hardware_run_times, "QPU"),
]:
    image = ax.imshow(
        density.T,
        aspect="auto",
        origin="lower",
        extent=[min(sweep_times), max(sweep_times), 0.5, 2.5],
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Run time ($\\mu s$)")
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Atom 1", "Atom 2"])

axs[0].set_ylabel("Site")
fig.colorbar(image, ax=axs, label="Rydberg density")
plt.show()
