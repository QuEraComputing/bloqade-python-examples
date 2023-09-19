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
# ## Introduction
# In this example we will show how to generate a Floquet protocol
# We will define the probotol using a python function and then
# use the Bloqade API to sample the function at certain intervals
# to make it compatible with the hardware, which only supports
# piecewise linear/constant functions. First let us start with 
# the imports.

# %%
from bloqade import start, cast, save, load
import os
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Define the program. 
# For the floquet protocol we keep
# a constant Rabi frequency but allow the detuning to vary sinusoidally.
# We do this by defining a smooth function for the detuning and then
# sampling it at certain intervals (in this case,
# the minimum hardware-supported time step). Note that the `sample` method
# will always sample at equal to or greater than the specified time step. 
# If the total time interval is not divisible by the time step, the last
# time step will be larger than the specified time step. Also note that
# the arguments of your function must be named arguments, e.g. no `*args`
# or `**kwargs`, because Bloqade will analyze the function signature to
# and generate variables for each argument.

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
# We assign values to the necessary variables and then run_async
# the program to both the emulator and
# actual hardware.

# %%
run_times = np.linspace(0.05, 3.0, 101)

floquet_job = floquet_program.assign(
    ramp_time=0.06,
    min_time_step=0.05,
    rabi_max=15,
    drive_amplitude=15,
    drive_frequency=15,
).batch_assign(run_time=run_times)

# %% [markdown]
# have to start the time at 0.05 because the hardware does not support
# anything less than that time step. We can now run_async the job to the
# emulator and hardware.

# %% [markdown]
# ## Run Emulator and Hardware
# To run the program on the emulator we can select the `braket` provider
# as a property of the `batch` object. Braket has its own emulator that
# we can use to run the program. To do this select `local_emulator` as
# the next option followed by the `run` method. Then we dump the results
# to a file so that we can use them later.

# %%
emu_filename = os.path.join(os.path.abspath(""), "data", "floquet-emulation.json")

if not os.path.isfile(emu_filename):
    emu_batch = floquet_job.braket.local_emulator().run(10000)
    save(emu_batch, emu_filename)

# %% [markdown]
# When running on the hardware we can use the `braket` provider as well.
# However, we will need to specify the `device` to run on. In this case
# we will use `Aquila` via the `aquila` method. Before that we must note 
# that because Aquila can support up to 256 atoms we need to make full use
# of the capabilities of the device. As we discussed in the Rabi example
# we can use the `parallelize` which will allow us to run multiple copies of
# the program in parallel using the full user provided area of Aquila. This 
# has to be put before the `braket` provider. Then we dump the results
# to a file so that we can use them later.

# %%
hardware_filename = os.path.join(os.path.abspath(""), "data", "floquet-job.json")

if not os.path.isfile(hardware_filename):
    batch = floquet_job.parallelize(24).braket.aquila().run_async(shots=50)
    save(batch, hardware_filename)

# %% [markdown]
# ## Plotting the Results
# The observables that we are interested in are the Rydberg population
# as a function of time. We can get this by first loading the results
# from the files that we saved earlier. Next each batch has a `report`
# method that will return a `Report` object. This object has a number
# of methods that are useful for different types of analysis. In this
# case we will use both the `list_param` and `bitstrings` methods. 
# To load the results we can use the `load` function from the `bloqade`

# %%
emu_batch = load(emu_filename)
hardware_batch = load(hardware_filename)
# hardware_batch.fetch()
# save(filename, hardware_batch)

# %% [markdown]
# Next we extract the run times and the Rydberg population from the
# report. We can then plot the results.

# %%

hardware_report = hardware_batch.report()
emulator_report = emu_batch.report()

times = emulator_report.list_param("run_time")
density = [1 - ele.mean() for ele in emulator_report.bitstrings()]
plt.plot(times, density, color="#878787", marker=".", label="emulation")

times = hardware_report.list_param("run_time")
density = [1 - ele.mean() for ele in hardware_report.bitstrings()]

plt.plot(times, density, color="#6437FF", linewidth=4, label="qpu")
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Rydberg population")
plt.legend()
plt.show()
