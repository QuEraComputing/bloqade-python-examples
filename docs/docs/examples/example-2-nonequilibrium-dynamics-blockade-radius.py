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

# %%


from bloqade import start, save_batch, load_batch
from bloqade.ir.location import Chain

import numpy as np
import matplotlib.pyplot as plt

import os

initial_geometry = Chain(2, "distance")
program_waveforms = (
    initial_geometry
    .rydberg
    .rabi
    .amplitude
    .uniform
    .piecewise_linear(durations=["ramp_time", "run_time", "ramp_time"], values=[0.0, "rabi_drive", "rabi_drive", 0.0])
    )
program_assigned_vars = program_waveforms.assign(ramp_time=0.06, rabi_drive=15, distance=8.5)
batch = program_assigned_vars.batch_assign(run_time = 0.05 * np.arange(31))

# send off to HW and Emulator
emu_batch = batch.braket.local_emulator().run(10000)

filename = os.path.join(os.path.abspath(""), "data", "nonequilibrium-dynamics-blockade.json")

if not os.path.isfile(filename):
    hardware_batch = batch.parallelize(24).braket.aquila().submit(shots=100)
    save_batch(filename, hardware_batch)

# plot data
emu_report = emu_batch.report()
emu_counts = emu_report.counts

hardware_batch = load_batch(filename)
# hardware_batch.fetch()
# save_batch(filename, hardware_batch)
hardware_report = hardware_batch.report()
hardware_counts = hardware_report.counts

def rydberg_state_probabilities(shot_counts):

    probabilities_dict = { "0": [], "1": [], "2": []}

    # iterate over each of the task results
    for task_result in shot_counts:
        # get total number of shots
        total_shots = sum(task_result.values())
        # get probability of each state
        probabilities_dict["0"].append(task_result.get("11",0) / total_shots)
        probabilities_dict["1"].append((task_result.get("10",0) + task_result.get("01", 0)) / total_shots)
        probabilities_dict["2"].append(task_result.get("00",0) / total_shots)

    return probabilities_dict

emu_probabilities = rydberg_state_probabilities(emu_counts)
hw_probabilities  = rydberg_state_probabilities(hardware_counts)

# plot 0, 1, and 2 rydberg state probabilities but in separate plots
figure, axs = plt.subplots(1, 3, figsize=(12,6), sharey=True)

run_times = 0.05 * np.arange(31)

axs[0].plot(run_times,emu_probabilities['0'],marker=".",color="#878787")
axs[0].plot(run_times,hw_probabilities['0'], color="#6437FF",linewidth=4)
axs[0].title.set_text("0 Rydberg State")

axs[1].plot(run_times,emu_probabilities['1'],marker=".",color="#878787")
axs[1].plot(run_times,hw_probabilities['1'],color="#6437FF",linewidth=4)
axs[1].title.set_text("1 Rydberg State")

axs[2].plot(run_times,emu_probabilities['2'],marker=".",color="#878787", label="emulation")
axs[2].plot(run_times,hw_probabilities['2'],color="#6437FF", linewidth=4, label="qpu")
axs[2].title.set_text("2 Rydberg State")

axs[0].set_ylabel("Probability")
axs[2].legend()

figure.show()