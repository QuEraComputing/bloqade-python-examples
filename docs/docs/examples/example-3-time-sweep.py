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
# # Whitepaper Example 3:

# %%
from bloqade import save, load
from bloqade.atom_arrangement import Chain
import numpy as np
import os
import matplotlib.pyplot as plt

n_atoms = 11
lattice_const = 6.1
min_time_step = 0.05

rabi_amplitude_values = [0.0, 15.8, 15.8, 0.0]
rabi_detuning_values = [-16.33, -16.33, 16.33, 16.33]
durations = [0.8, "sweep_time", 0.8]

time_sweep_z2_prog = (
    Chain(n_atoms, lattice_const)
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude_values)
    .detuning.uniform.piecewise_linear(durations, rabi_detuning_values)
)

time_sweep_z2_job = time_sweep_z2_prog.batch_assign(
    sweep_time=np.linspace(min_time_step, 2.4, 20)
)  # starting at 0.0 not feasible, just use min_time_step

# run_async to emulator

emu_filename = os.path.join(os.path.abspath(""), "data", "time-sweep-emulation.json")
if not os.path.isfile(emu_filename):
    emu_future = time_sweep_z2_job.braket.local_emulator().run_async(shots=10000)
    save(emu_future, emu_filename)

filename = os.path.join(os.path.abspath(""), "data", "time-sweep-job.json")
if not os.path.isfile(filename):
    future = time_sweep_z2_job.parallelize(24).braket.aquila().run_async(shots=10000)
    save(future, filename)
    
# %% [markdown]



# %% 

def get_z2_probabilities(report):
    z2_probabilities = []

    for count in report.counts:
        z2_probability = count["01010101010"] / sum(list(count.values()))
        z2_probabilities.append(z2_probability)

    return z2_probabilities



# retrieve results from HW
emu_batch = load(emu_filename)
hardware_batch = load(filename)
# hardware_batch = hardware_batch.fetch()
# save(hardware_batch, filename)

emu_report = emu_batch.report()
hardware_report = hardware_batch.report()
emu_probabilities = get_z2_probabilities(emu_report)
hardware_probabilities = get_z2_probabilities(emu_report)

emu_sweep_times = emu_report.list_param("sweep_time")
hardware_sweep_times = hardware_report.list_param("sweep_time")


plt.plot(emu_sweep_times, emu_probabilities, label="Emulator")
plt.plot(hardware_sweep_times, hardware_probabilities, label="Hardware")
plt.show()

