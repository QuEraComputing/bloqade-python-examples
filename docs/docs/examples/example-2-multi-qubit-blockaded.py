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
from bloqade.ir.location import Chain, Square
import numpy as np
import matplotlib.pyplot as plt

import os

def multi_qubit_blockade_batch(n_atoms):

    distance = 4.0 
    inv_sqrt_2_rounded = 2.6

    geometries = \
    { 1: Chain(1), 
      2: Chain(2, distance), 
      3: start.add_positions([(-inv_sqrt_2_rounded, 0.0), (inv_sqrt_2_rounded, 0.0), (0, distance)]),
      4: Square(2, distance), 
      7: start.add_positions([(0, 0), (distance, 0), (-0.5*distance, distance), (0.5*distance, distance), (1.5*distance, distance), (0, 2*distance), (distance, 2*distance)])
    }
    if n_atoms not in geometries:
        raise ValueError("natoms must be 1,2,3,4, or 7")
    
    program_base = geometries[n_atoms]

    full_program = program_base.rydberg.rabi.amplitude
    full_program = full_program.uniform.piecewise_linear(durations = ["ramp_time","run_time", "ramp_time"], values = [0.0, "rabi_drive", "rabi_drive", 0.0])
    full_program =full_program.assign(ramp_time = 0.06, rabi_drive = 5).batch_assign(run_time = 0.05 * np.arange(21))
    
    return full_program

# submit to Emulator and HW

generated_batch = multi_qubit_blockade_batch(7)

emu_batch = generated_batch.braket.local_emulator().run(10000)

filename = os.path.join(os.path.abspath(""), "data", "multi-qubit-blockaded.json")

if not os.path.isfile(filename):
    hardware_batch = generated_batch.parallelize(24).braket.aquila().submit(shots=100)
    save_batch(filename, hardware_batch)

# get emulation report and calculate the sum of Rydberg densities
emu_report = emu_batch.report()
emu_densities = emu_report.rydberg_densities()
emu_densities_summed = emu_densities.sum(axis=1)

# get hardware report and number of shots per each state
hardware_batch = load_batch(filename)
# hardware_batch.fetch()
# save_batch(filename, hardware_batch)
hardware_report = hardware_batch.report()
hardware_densities = hardware_report.rydberg_densities()
hardware_densities_summed = hardware_densities.sum(axis=1)

# plot results
## Want to show the sum of Rydberg densities

fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Sum of Rydberg Densities")
# emulation
ax.plot(0.05 * np.arange(21), emu_densities_summed, label="Emulation", color="#878787")
# hardware
ax.plot(0.05 * np.arange(21), hardware_densities_summed, label="Hardware", color="#6437FF")