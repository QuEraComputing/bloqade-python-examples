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
# In this example we show how to create the Striated Phase on a small 2D square
# lattice of atoms. The protocol is similar in spirit to the 1D Z2 preparation:
# start with negative detuning, turn on the Rabi coupling, sweep to positive
# detuning, and let the Rydberg blockade select an ordered pattern. In two
# dimensions the ordering is easier to understand when we look at the spatial
# density pattern rather than only at raw bitstring counts.

# %% [markdown]
# You might notice that the tools we need to import are
# a lot shorter than prior instances. This is because
# we're taking advantage of bloqade Python's built-in
# visualization capabilities instead of crafting
# a new plot with matplotlib.
# %%
import os

import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import output_notebook
from bloqade.analog import load, save
from bloqade.analog.atom_arrangement import Square

if not os.path.isdir("data"):
    os.mkdir("data")

# This tells Bokeh to display output in the notebook
# versus opening a browser window
output_notebook()

# %% [markdown]
# ## Program Definition
# We define a program where our geometry is a square lattice of 3x3 atoms. Notice that
# unlike the 1D Z2 state preparation example the detuning now ramps to a higher value
# and the atoms are closer together.
# %%
# Have atoms separated by 5.9 micrometers
L = 3
lattice_spacing = 5.9

rabi_amplitude_values = [0.0, 15.8, 15.8, 0.0]
rabi_detuning_values = [-16.33, -16.33, "delta_end", "delta_end"]
durations = [0.8, "sweep_time", 0.8]

prog = (
    Square(L, lattice_spacing=lattice_spacing)
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude_values)
    .detuning.uniform.piecewise_linear(durations, rabi_detuning_values)
)

batch = prog.assign(delta_end=42.66, sweep_time=2.4)

# %% [markdown]
# Before submitting the program, it is useful to preview both the geometry and the
# pulse schedule. The nearest-neighbor spacing is chosen so that blockade effects
# compete strongly with the positive final detuning, while the square geometry lets
# us look for row- or column-like striations in the measured Rydberg density.

# %%
site_x, site_y = np.meshgrid(
    np.arange(L) * lattice_spacing,
    np.arange(L) * lattice_spacing,
)
site_indices = np.arange(L * L)

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.scatter(site_x.ravel(), site_y.ravel(), s=160, color="#6437FF")
for site_index, x_coord, y_coord in zip(site_indices, site_x.ravel(), site_y.ravel()):
    ax.annotate(
        str(site_index),
        (x_coord, y_coord),
        ha="center",
        va="center",
        color="white",
        fontsize=9,
    )
ax.set_aspect("equal")
ax.set_xlabel(r"x ($\mu m$)")
ax.set_ylabel(r"y ($\mu m$)")
ax.set_title("3 x 3 square lattice")
plt.show()

# %%
preview_times = np.concatenate([[0.0], np.cumsum([0.8, 2.4, 0.8])])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    preview_times,
    rabi_amplitude_values,
    marker=".",
    color="#6437FF",
    label="Rabi amplitude",
)
ax.plot(
    preview_times,
    [-16.33, -16.33, 42.66, 42.66],
    marker=".",
    color="#C2477F",
    label="Detuning",
)
ax.axhline(0, color="#878787", linestyle="--", linewidth=1)
ax.set_xlabel(r"time ($\mu s$)")
ax.set_ylabel(r"angular frequency (rad/$\mu s$)")
ax.set_title("2D striated-state preparation pulse")
ax.legend()
plt.show()

# %% [markdown]
# ## Submitting to Emulator and Hardware
# Just as in prior examples, we submit our program to both hardware and the emulator and
# save the intermediate data in files for convenient fetching when the results are ready from hardware,
# as well as avoiding having to repeat emulation runs for the purposes of analysis.

# Considering how small a 3 x 3 lattice of atoms is relative to machine capabilities,
# we also take advantage of parallelization to duplicate the geometry and get more
# data per shot when submitting to Hardware.

#
# <div class="admonition danger">
# <p class="admonition-title">Hardware Execution Cost</p>
# <p>
#
# For this particular program, 1 task is generated with 100 shots, amounting to
#  __USD \\$1.30__ on AWS Braket.
#
# </p>
# </div>

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
emu_report = emu_batch.report()
hardware_report = hardware_batch.report()

emu_report.show()

# %% [markdown]
# Just as before, we let Bloqade generate a `report` which contains all the results in
# easy to digest format but we invoke the `.show()` method of our report which us
# easily get an idea of the results of our experiment with interactive plots.

# The plot that most interests us is the one on the right under the "Rydberg Density" section.

# %%
hardware_report.show()

# %% [markdown]
# Considering Bloqade's goal of a uniform visualization pipeline, we can get the same
# ability for results from hardware. Note that we can confirm the program does what it's
# supposed to as results from emulation agree with those from hardware quite well.

# %% [markdown]
# The interactive reports above are convenient, but for a tutorial it also helps to
# place the emulator and hardware densities on the same color scale. The heatmaps
# below reshape the site-resolved Rydberg densities back onto the 3 x 3 lattice and
# add a QPU-minus-emulator residual panel so the agreement is visible at a glance.


# %%
def density_grid(report):
    densities = np.asarray(report.rydberg_densities(), dtype=float).reshape(-1)
    if densities.size != L * L:
        raise ValueError(f"Expected {L * L} densities, received {densities.size}")
    return densities.reshape(L, L)


emu_density_grid = density_grid(emu_report)
hardware_density_grid = density_grid(hardware_report)
density_residual = hardware_density_grid - emu_density_grid
residual_limit = float(np.max(np.abs(density_residual)))
if residual_limit == 0:
    residual_limit = 1e-9

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
for ax, density_grid_values, title in [
    (axes[0], emu_density_grid, "Emulator"),
    (axes[1], hardware_density_grid, "Aquila QPU"),
]:
    image = ax.imshow(density_grid_values, origin="lower", vmin=0, vmax=1, cmap="magma")
    ax.set_title(title)
    ax.set_xticks(np.arange(L))
    ax.set_yticks(np.arange(L))
    ax.set_xlabel("lattice column")
axes[0].set_ylabel("lattice row")
fig.colorbar(image, ax=axes[:2], shrink=0.85, label="Rydberg density")

residual_image = axes[2].imshow(
    density_residual,
    origin="lower",
    vmin=-residual_limit,
    vmax=residual_limit,
    cmap="coolwarm",
)
axes[2].set_title("QPU - emulator")
axes[2].set_xticks(np.arange(L))
axes[2].set_yticks(np.arange(L))
axes[2].set_xlabel("lattice column")
fig.colorbar(residual_image, ax=axes[2], shrink=0.85, label="Density difference")
plt.show()

# %% [markdown]
# We can also compress the density map into row and column averages. This provides a
# quick diagnostic for whether the ordered state is forming a striated pattern along
# one lattice direction and whether the same qualitative contrast appears in the
# hardware data.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
for ax, reducer, title in [
    (axes[0], lambda grid: grid.mean(axis=1), "Row-averaged density"),
    (axes[1], lambda grid: grid.mean(axis=0), "Column-averaged density"),
]:
    positions = np.arange(L)
    width = 0.36
    ax.bar(
        positions - width / 2,
        reducer(emu_density_grid),
        width,
        color="#878787",
        label="Emulator",
    )
    ax.bar(
        positions + width / 2,
        reducer(hardware_density_grid),
        width,
        color="#6437FF",
        label="QPU",
    )
    ax.set_xticks(positions)
    ax.set_xlabel("lattice index")
    ax.set_title(title)
axes[0].set_ylabel("mean Rydberg density")
axes[1].legend()
plt.show()
