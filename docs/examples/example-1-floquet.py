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
# # Floquet Protocol

# %%
from bloqade import start, cast, save_batch, load_batch
from decimal import Decimal
import numpy as np
import os

# %% [markdown]
# Define the program. For the floquet protocol we keep
# a constant Rabi frequency but allow the detuning to vary sinusoidally.
#
# We do this by defining a smooth function for the detuning and then
# sampling it at certain intervals (in this case,
# the minimum hardware-supported time step)

# %%

min_time_step = 0.05

durations = cast(["ramp_time", "run_time", "ramp_time"])


def detuning_wf(t, drive_amplitude, drive_frequency):
    return drive_amplitude * np.sin(drive_frequency * t)


floquet_program = (
    start.add_position((0, 0))
    .rydberg.rabi.amplitude.uniform.piecewise_linear(
        durations, [0, "rabi_max", "rabi_max", 0]
    )
    .detuning.uniform.fn(detuning_wf, sum(durations))
    .sample("min_time_step", "linear")
)

# %% [markdown]
# We assign values to the necessary variables and then submit
# the program to both the emulator and
# actual hardware.

# %%
n_steps = 100
max_time = Decimal("3.0")
dt = (max_time - Decimal("0.05")) / n_steps
run_times = [Decimal("0.05") + dt * i for i in range(101)]

floquet_job = floquet_program.assign(
    ramp_time=0.06,
    min_time_step=0.05,
    rabi_max=15,
    drive_amplitude=15,
    drive_frequency=15,
).batch_assign(run_time=run_times)
# have to start the time at 0.05 considering 0.03 (generated if we start at 0.0)
# is considered too small by validation

# submit to emulator
emu_batch = floquet_job.braket.local_emulator().run(shots=10000)

# submit to HW
filename = os.path.join(os.path.dirname(__file__), "data", "floquet-job.json")

if not os.path.isfile(filename):
    batch = floquet_job.parallelize(24).braket.aquila().submit(shots=50)
    save_batch(filename, batch)

# %% [markdown]
# Load JSON and pull results from Braket
filename = os.path.join(os.path.dirname(__file__), "data", "floquet-job.json")
hardware_batch = load_batch(filename)
hardware_batch.fetch()
save_batch(filename, hardware_batch)

# %% [markdown]
# We can now plot the results from the hardware and emulation together.

# %%
import matplotlib.pyplot as plt

filename = os.path.join(os.path.dirname(__file__), "data", "floquet-job.json")

hardware_report = hardware_batch.fetch().report()
emulator_report = emu_batch.report()

times = emulator_report.list_param("run_time")
density = [1 - ele.mean() for ele in emulator_report.bitstrings()]
plt.plot(times, density)

times = hardware_report.list_param("run_time")
density = [1 - ele.mean() for ele in hardware_report.bitstrings()]

plt.plot(times, density)
plt.show()
