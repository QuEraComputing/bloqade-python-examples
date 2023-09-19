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
# # Whitepaper Example 3: 2D Ordered State

# %%
from bloqade.atom_arrangement import Square
from bokeh.io import output_notebook

output_notebook()


# %%
L = 3
lattice_const = 7

rabi_amplitude_values = [0.0, 15.8, 15.8, 0.0]
rabi_detuning_values = [-16.33, -16.33, "delta_end", "delta_end"]
durations = [0.8, "sweep_time", 0.8]

prog = (
    Square(L, lattice_const)
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude_values)
    .detuning.uniform.piecewise_linear(durations, rabi_detuning_values)
)

batch = prog.assign(delta_end=42.66, sweep_time=1.1)


# %%
result = batch.braket.local_emulator().run(shots=1000)

result.report().show()

