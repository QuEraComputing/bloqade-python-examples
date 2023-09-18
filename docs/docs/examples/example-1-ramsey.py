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
# # Ramsey Protocol
# ## Introduction
# In this example we show how to use Bloqade to emulate a
# Ramsey protocol as well as run it on hardware.

# %%
from bloqade import start, cast, save, load
from decimal import Decimal
import os
import numpy as np

# %% [markdown]

# define program with one atom, with constant detuning but variable Rabi frequency,
# where an initial pi/2 pulse is applied, followed by some time gap and a -pi/2 pulse

# %%
plateau_time = (np.pi / 2 - 0.625) / 12.5
wf_durations = cast([0.05, plateau_time, 0.05, "run_time", 0.05, plateau_time, 0.05])
rabi_wf_values = [0.0, 12.5, 12.5, 0.0] * 2  # repeat values twice

ramsey_program = (
    start.add_position([0, 0])
    .rydberg.rabi.amplitude.uniform.piecewise_linear(wf_durations, rabi_wf_values)
    .detuning.uniform.constant(10.5, sum(wf_durations))
)

# %% [markdown]
# Assign values to the variables in the program,
# allowing `run_time` (time gap between the two pi/2 pulses)
# to sweep across a range of values.

# %%
n_steps = 100
max_time = Decimal("3.0")
dt = (max_time - Decimal("0.05")) / n_steps
run_times = [Decimal("0.05") + dt * i for i in range(101)]

ramsey_job = ramsey_program.batch_assign(run_time=run_times)

# %% [markdown]
# Run the program in emulation, obtaining a report
# object. For each possible set of variable values
# to simulate (in this case, centered around the
# `run_time` variable), let the task have 10000 shots.

# %%
emu_batch = ramsey_job.braket.local_emulator().run(shots=10000)

# %% [markdown]
# Submit the same program to hardware,
# this time using `.parallelize` to make a copy of the original geometry
# (a single atom) that fills the FOV (Field-of-View Space), with at least
# 24 micrometers of distance between each atom.
#
# Unlike the emulation above, we only let each task
# run with 100 shots. A collection of tasks is known as a
# "Job" in Bloqade and jobs can be saved in JSON format
# so you can reload them later (a necessity considering
# how long it may take for the machine to handle tasks in the queue)

# %%
filename = os.path.join(os.path.abspath(""), "data", "ramsey-job.json")
if not os.path.isfile(filename):
    batch = ramsey_job.parallelize(24).braket.aquila().submit(shots=100)
    save(filename, batch)
# %% [markdown]
# Load JSON and pull results from Braket

# %%
filename = os.path.join(os.path.abspath(""), "data", "ramsey-job.json")
hardware_batch = load(filename)
# hardware_batch.fetch()
#save(filename, hardware_batch)

# %% [markdown]
# We can now plot the results from the hardware and emulation together.

# %%
import matplotlib.pyplot as plt

hardware_report = hardware_batch.report()
emulator_report = emu_batch.report()

times = emulator_report.list_param("run_time")
density = [1 - ele.mean() for ele in emulator_report.bitstrings()]
plt.plot(times, density)

times = hardware_report.list_param("run_time")
density = [1 - ele.mean() for ele in hardware_report.bitstrings()]

plt.plot(times, density)
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Rydberg population")
plt.show()
