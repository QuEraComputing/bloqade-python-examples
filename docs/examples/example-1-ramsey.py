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
# <a class=md-button href="example-1-ramsey.py" download> Download Script </a>
# <a class=md-button href="../../assets/data/ramsey-job.json" download> Download Job </a>
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
# # Single Qubit Ramsey Protocol
# ## Introduction
# In this example we show how to use Bloqade to emulate a Ramsey protocol as well as
# run it on hardware. We will define a Ramsey protocol as a sequence of two $\pi/2$
# pulses separated by a variable time gap $\tau$. These protocols are used to measure the
# coherence time of a qubit. In practice, the Rabi frequency has to start and end at
# 0.0, so we will use a piecewise linear function to ramp up and down the Rabi
# frequency.
#
# The first pulse prepares a coherent superposition between the ground and Rydberg
# states. During the variable gap, the state accumulates phase relative to the drive.
# The second pulse maps that phase back into a population difference, which produces
# the Ramsey fringe plotted at the end of the example.

# %%
import os
from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from bloqade.analog import cast, load, save, start

if not os.path.isdir("data"):
    os.mkdir("data")

# %% [markdown]
# ## Define the program.
# define program with one atom, with constant detuning but variable Rabi frequency,
# where an initial pi/2 pulse is applied, followed by some time gap and a -pi/2 pulse.
# Note that the plateau time was chosen such that the area under the curve is pi/2 given
# given the constraint on how fast the Rabi frequency can change as well as the minimum
# allowed time step.

# %%
ramp_time = 0.05
rabi_ampl = 12.5
detuning_value = 10.5
plateau_time = (np.pi / 2 - 0.625) / rabi_ampl
wf_durations = cast(
    [
        ramp_time,
        plateau_time,
        ramp_time,
        "run_time",
        ramp_time,
        plateau_time,
        ramp_time,
    ]
)
rabi_wf_values = [0.0, rabi_ampl, rabi_ampl, 0.0] * 2  # repeat values twice

ramsey_program = (
    start.add_position((0, 0))
    .rydberg.rabi.amplitude.uniform.piecewise_linear(wf_durations, rabi_wf_values)
    .detuning.uniform.constant(detuning_value, sum(wf_durations))
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
# The plot below shows the pulse schedule for the longest wait time in the sweep. The
# two short Rabi pulses are the Ramsey $\pi/2$ rotations, while the shaded middle region
# is the free-evolution interval whose duration is scanned. The detuning stays constant
# throughout the sequence and controls the phase accumulation during that interval.

# %%
example_gap = float(run_times[-1])
pulse_durations = np.array(
    [
        ramp_time,
        plateau_time,
        ramp_time,
        example_gap,
        ramp_time,
        plateau_time,
        ramp_time,
    ],
    dtype=float,
)
pulse_times = np.concatenate(([0.0], np.cumsum(pulse_durations)))
pulse_amplitudes = np.array(
    [0.0, rabi_ampl, rabi_ampl, 0.0, 0.0, rabi_ampl, rabi_ampl, 0.0]
)
detuning_values = np.full_like(pulse_times, detuning_value, dtype=float)

first_pulse_end = 2 * ramp_time + plateau_time
second_pulse_start = first_pulse_end + example_gap

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(
    pulse_times,
    pulse_amplitudes,
    color="#C8447C",
    linewidth=2,
    label="Rabi amplitude",
)
ax.plot(
    pulse_times,
    detuning_values,
    color="#878787",
    linestyle="--",
    label="Detuning",
)
ax.axvspan(
    first_pulse_end,
    second_pulse_start,
    color="#E9E1FF",
    alpha=0.45,
    label="Free evolution",
)
ax.set_xlabel("Time ($\mu s$)")
ax.set_ylabel("Angular frequency (rad/$\mu s$)")
ax.set_title("Single-qubit Ramsey pulse schedule")
ax.legend()
plt.show()

# %% [markdown]
# ## Run Emulation and Hardware
# Like in the first tutorial, we will run the program on the emulator and hardware.
# Note that for the hardware we will use the `parallelize` method to run multiple
# copies of the program in parallel. For more information about this process, see the
# first tutorial.
#
# <div class="admonition danger">
# <p class="admonition-title">Hardware Execution Cost</p>
# <p>
#
# For this particular program, 101 tasks are generated with each task having 100 shots, amounting to
#  __USD \\$131.30__ on AWS Braket.
#
# </p>
# </div>

# %%
emu_filename = os.path.join(os.path.abspath(""), "data", "ramsey-emulation.json")

if not os.path.isfile(emu_filename):
    emu_batch = ramsey_job.bloqade.python().run(10000)
    save(emu_batch, emu_filename)

hardware_filename = os.path.join(os.path.abspath(""), "data", "ramsey-job.json")
if not os.path.isfile(hardware_filename):
    batch = ramsey_job.parallelize(24).braket.aquila().run_async(shots=100)
    save(batch, hardware_filename)

# %% [markdown]
# ## Plot the results
# Exactly like in the Rabi Oscillation example, we can now plot the results from the
# hardware and emulation together. Again we will use the `report` to calculate the mean
# Rydberg population for each run, and then plot the results.
#
# The horizontal axis is the free-evolution gap between the two pulses. Oscillations in
# the Rydberg population are Ramsey fringes: changing the gap changes the phase that the
# second pulse converts back into a measurable Rydberg-state probability.
#
# first we load the results from the emulation and hardware.
# %%
emu_batch = load(emu_filename)
hardware_batch = load(hardware_filename)
# hardware_batch.fetch()
# save(filename, hardware_batch)

# %% [markdown]
# Next we can calculate the Rydberg population for each run and plot the results.

# %%

hardware_report = hardware_batch.report()
emulator_report = emu_batch.report()

emu_times = np.array(emulator_report.list_param("run_time"), dtype=float)
emu_density = [1 - ele.mean() for ele in emulator_report.bitstrings()]

hardware_times = np.array(hardware_report.list_param("run_time"), dtype=float)
hardware_density = [1 - ele.mean() for ele in hardware_report.bitstrings()]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(emu_times, emu_density, color="#878787", marker=".", label="Emulator")
ax.plot(hardware_times, hardware_density, color="#6437FF", linewidth=4, label="QPU")
ax.set_xlabel("Free-evolution time ($\mu s$)")
ax.set_ylabel("Rydberg population")
ax.set_title("Ramsey fringes from variable phase accumulation")
ax.legend()
plt.show()
