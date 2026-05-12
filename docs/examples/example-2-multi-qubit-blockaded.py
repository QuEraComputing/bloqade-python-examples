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
# <a class=md-button href="example-2-multi-qubit-blockaded.py" download> Download Script </a>
# <a class=md-button href="../../assets/data/multi-qubit-blockaded-job.json" download> Download Job </a>
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
# # Multi-qubit Blockaded Rabi Oscillations
# ## Introduction
# In this tutorial we will show you how to compose geometries with pulse sequences to
# perform multi-qubit blockaded Rabi oscillations. The Physics here is described in
# detail in the [whitepaper](https://arxiv.org/abs/2306.11727). But in short, we can
# use the Rydberg blockade to change the effective Rabi frequency of the entire system
# by adding more atoms to the cluster.
# %%
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from bloqade.analog import load, save, start
from bloqade.analog.atom_arrangement import Chain, Square

if not os.path.isdir("data"):
    os.mkdir("data")

ramp_time = 0.06
rabi_drive = 5.0
run_times = 0.05 * np.arange(21)

# %% [markdown]
# ## Defining the Geometry
# We will start by defining the geometry of the atoms. The idea here is to cluster
# the atoms so that they are all blockaded from each other. Using a combination of the
# `Chain` and `Square` classes, as a base, one can add additional atoms to the geometry
# using the `add_position` method. This method takes a list of tuples, or a single
# tuple, of the form `(x,y)` where `x` and `y` are the coordinates of the atom in units
# of the lattice constant.

# %%

distance = 4.0
inv_sqrt_2_rounded = 2.6
seven_atom_positions = [
    (0, 0),
    (distance, 0),
    (-0.5 * distance, distance),
    (0.5 * distance, distance),
    (1.5 * distance, distance),
    (0, 2 * distance),
    (distance, 2 * distance),
]

geometries = {
    1: Chain(1),
    2: Chain(2, lattice_spacing=distance),
    3: start.add_position(
        [(-inv_sqrt_2_rounded, 0.0), (inv_sqrt_2_rounded, 0.0), (0, distance)]
    ),
    4: Square(2, lattice_spacing=distance),
    7: start.add_position(seven_atom_positions),
}

# %% [markdown]
# The seven-atom cluster is intentionally compact. For resonant driving, the blockade
# radius is approximately $(C_6 / \Omega)^{1/6}$, so the $5$ MHz drive used below gives
# a blockade radius larger than the maximum pair distance in this geometry. This keeps
# the whole cluster in the collective blockade regime.

# %%
aquila_c6 = 2 * np.pi * 862690
blockade_radius = (aquila_c6 / rabi_drive) ** (1 / 6)
cluster_positions = np.array(seven_atom_positions)
pairwise_distances = np.linalg.norm(
    cluster_positions[:, None, :] - cluster_positions[None, :, :], axis=-1
)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], color="#6437FF", zorder=3)
for atom_index, (x_position, y_position) in enumerate(cluster_positions):
    ax.annotate(
        str(atom_index),
        (x_position, y_position),
        xytext=(5, 5),
        textcoords="offset points",
    )

blockade_disk = Circle(
    cluster_positions[0],
    blockade_radius,
    color="#C8447C",
    alpha=0.12,
    label=f"Blockade radius {blockade_radius:.1f} $\\mu$m",
)
ax.add_patch(blockade_disk)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"x position ($\mu m$)")
ax.set_ylabel(r"y position ($\mu m$)")
ax.set_title("Seven-atom blockaded cluster")
ax.legend(loc="upper right")
ax.set_xlim(-blockade_radius - 1, blockade_radius + 1)
ax.set_ylim(-blockade_radius - 1, blockade_radius + 1)
plt.show()

print(f"Maximum atom-pair distance: {pairwise_distances.max():.2f} micrometers")

# %% [markdown]
# ## Defining the Pulse Sequence
# Next, we will define the pulse sequence. We start from the `start` object, which is
# an empty list of atom locations. In this case, we do not need atoms to build the pulse
# sequence, but to extract the sequence, we need to call the `parse_sequence` method.
# This creates a `Sequence` object that we can apply to multiple geometries.
# %%
sequence = start.rydberg.rabi.amplitude.uniform.piecewise_linear(
    durations=["ramp_time", "run_time", "ramp_time"],
    values=[0.0, "rabi_drive", "rabi_drive", 0.0],
).parse_sequence()
# %% [markdown]
# The drive has a flat-top profile: a short turn-on ramp, a variable hold time, and a
# matching turn-off ramp. Since every atom in the cluster is inside the blockade radius,
# the pulse couples the collective ground state to the symmetric one-excitation state
# instead of driving each atom independently. Increasing the hold time lets us resolve
# that collective Rabi oscillation.
#
# The preview below shows one member of the sweep. Only the hold time changes from task
# to task; the ramp time and peak Rabi frequency stay fixed.

# %%
pulse_preview_run_time = run_times[10]

pulse_times = np.array(
    [
        0.0,
        ramp_time,
        ramp_time + pulse_preview_run_time,
        2 * ramp_time + pulse_preview_run_time,
    ]
)
pulse_values = np.array([0.0, rabi_drive, rabi_drive, 0.0])

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(pulse_times, pulse_values, color="#6437FF", marker="o")
ax.fill_between(pulse_times, pulse_values, color="#6437FF", alpha=0.15)
ax.set_xlabel(r"Time ($\mu s$)")
ax.set_ylabel("Rabi amplitude (MHz)")
ax.set_title("Example blockaded Rabi pulse")
ax.set_ylim(bottom=0)
plt.show()

# %% [markdown]

# ## Defining the Program
# Now, all that is left to do is to compose the geometry and the Pulse sequence into a
# fully defined program. We can do this by calling the `apply` method on the geometry
# and passing in the sequence. This method will return an object that can then be
# assigned parameters.
# %%
batch = (
    geometries[7]
    .apply(sequence)
    .assign(ramp_time=ramp_time, rabi_drive=rabi_drive)
    .batch_assign(run_time=run_times)
)

# %% [markdown]
# ## Run Emulator and Hardware
# Again, we run the program on the emulator and Aquila and save the results to a file
# so we can use them later.
#
# <div class="admonition danger">
# <p class="admonition-title">Hardware Execution Cost</p>
# <p>
#
# For this particular program, 21 tasks are generated with each task having 100 shots, amounting to
#  __USD \\$27.30__ on AWS Braket.
#
# </p>
# </div>
# %%

emu_filename = os.path.join(
    os.path.abspath(""), "data", "multi-qubit-blockaded-emulation.json"
)

if not os.path.isfile(emu_filename):
    emu_batch = batch.bloqade.python().run(10000, interaction_picture=True)
    save(emu_batch, emu_filename)

filename = os.path.join(os.path.abspath(""), "data", "multi-qubit-blockaded-job.json")

if not os.path.isfile(filename):
    hardware_batch = batch.parallelize(24).braket.aquila().run_async(shots=100)
    save(hardware_batch, filename)

# %% [markdown]
# ## Plotting the Results
# First, we load the results from the file.


# %%
emu_batch = load(emu_filename)
hardware_batch = load(filename)
# hardware_batch.fetch()
# save(filename, hardware_batch)

# %% [markdown]
# The quantity of interest here is the total Rydberg density of the cluster defined as
# the sum of the Rydberg densities of each atom. We can extract this from the results
# and plot it as a function of time. We will do this for both the emulator and the
# hardware. We can use the `rydberg_densities` function to extract the densities from
# the `Report` of the `batch` object.

# %%

emu_report = emu_batch.report()
emu_densities = emu_report.rydberg_densities()
emu_densities_summed = emu_densities.sum(axis=1)

hardware_report = hardware_batch.report()
hardware_densities = hardware_report.rydberg_densities()
hardware_densities_summed = hardware_densities.sum(axis=1)


emu_run_times = emu_report.list_param("run_time")
hw_run_times = hardware_report.list_param("run_time")

fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Sum of Rydberg Densities")
# emulation
ax.plot(emu_run_times, emu_densities_summed, label="Emulator", color="#878787")
# hardware
ax.plot(hw_run_times, hardware_densities_summed, label="QPU", color="#6437FF")
ax.axhline(
    1.0,
    color="#C8447C",
    linestyle="--",
    linewidth=1,
    label="One-excitation blockade limit",
)
ax.legend()
ax.set_xlabel(r"Time ($\mu s$)")
ax.set_ylabel("Sum of Rydberg Densities")
plt.show()

# %% [markdown]
# A summed density makes the collective oscillation easy to see, but it hides how that
# density is shared across the atoms. In the blockaded regime, simultaneous neighboring
# excitations are suppressed and the one-excitation component should be spread over the
# cluster rather than localized on one atom. A per-site heatmap gives a quick check that
# the response remains spatially symmetric while the total excitation oscillates.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, density_frame, run_time_values, title in [
    (axes[0], emu_densities, emu_run_times, "Emulator"),
    (axes[1], hardware_densities, hw_run_times, "QPU"),
]:
    run_time_values = np.asarray(run_time_values, dtype=float)
    time_step = run_time_values[1] - run_time_values[0]
    extent = [
        run_time_values[0] - time_step / 2,
        run_time_values[-1] + time_step / 2,
        -0.5,
        density_frame.shape[1] - 0.5,
    ]

    image = ax.imshow(
        density_frame.values.T,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        extent=extent,
        cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel(r"Run time ($\mu s$)")
    ax.set_yticks(np.arange(density_frame.shape[1]))
    ax.set_yticklabels([str(atom) for atom in density_frame.columns])

axes[0].set_ylabel("Atom index")
fig.colorbar(image, ax=axes, shrink=0.8, label="Rydberg density")
plt.show()
